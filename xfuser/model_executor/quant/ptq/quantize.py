"""Quantize Wan weights from saved Hessians into an MXFP4 PTQ artifact."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .artifact import QuantizedLinear, config_hash, write_artifact
from .gptq_mxfp4 import GPTQConfig, gptq_quantize_mxfp4, prepare_hessian
from .observers import wan_block_hessian_groups


def _checkpoint_map(model: str, revision: str, component: str) -> dict[str, str]:
    index = hf_hub_download(
        repo_id=model,
        filename=f"{component}/diffusion_pytorch_model.safetensors.index.json",
        revision=revision,
        local_files_only=True,
    )
    raw_map = json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"]
    resolved = {}
    paths = {}
    for key, filename in raw_map.items():
        if filename not in paths:
            paths[filename] = hf_hub_download(
                repo_id=model,
                filename=f"{component}/{filename}",
                revision=revision,
                local_files_only=True,
            )
        resolved[key] = paths[filename]
    return resolved


def _read_weight(weight_map: dict[str, str], key: str, device: str) -> torch.Tensor:
    path = weight_map.get(key)
    if path is None:
        raise KeyError(f"checkpoint has no tensor {key}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device)


def quantize_calibration(
    calibration_dir: Path,
    output_root: Path,
    *,
    blocks: tuple[int, ...],
    groups: frozenset[str] | None,
    config: GPTQConfig,
) -> Path:
    record = json.loads((calibration_dir / "calib_record.json").read_text(encoding="utf-8"))
    settings = record["settings"]
    model = settings["model"]
    revision = settings["model_revision"]
    calib_id = record["calib_id"]
    device = f"cuda:{torch.cuda.current_device()}"
    linears: dict[str, QuantizedLinear] = {}
    reports = []

    for component in ("transformer", "transformer_2"):
        hessians = torch.load(
            calibration_dir / f"{component}_hessians.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        weight_map = _checkpoint_map(model, revision, component)
        for block_index in blocks:
            for group in wan_block_hessian_groups(block_index):
                short_group = group.name.rsplit(".", 1)[-1]
                if groups is not None and short_group not in groups:
                    continue
                hessian_cpu = hessians.get(group.name)
                if hessian_cpu is None:
                    raise KeyError(f"calibration has no Hessian {component}.{group.name}")
                hessian = hessian_cpu.to(device)
                prepared = prepare_hessian(hessian, config)
                for module_name in group.linears:
                    started_at = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    weight = _read_weight(
                        weight_map,
                        f"{module_name}.weight",
                        device,
                    )
                    result = gptq_quantize_mxfp4(
                        weight,
                        hessian,
                        config,
                        prepared=prepared,
                    )
                    full_name = f"{component}.{module_name}"
                    linears[full_name] = QuantizedLinear(
                        packed=result.packed.cpu(),
                        scale=result.scale.cpu(),
                    )
                    torch.cuda.synchronize()
                    reports.append(
                        {
                            "layer": full_name,
                            **result.report,
                            "runtime_seconds": time.perf_counter() - started_at,
                            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                        }
                    )
                    del weight, result
                del prepared, hessian
                torch.cuda.empty_cache()

    gptq_payload = {
        "percdamp": config.percdamp,
        "max_cholesky_retries": config.max_cholesky_retries,
        "blocks": list(blocks),
        "groups": sorted(groups) if groups is not None else "all",
    }
    digest = config_hash(gptq_payload)[:12]
    artifact_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{digest}"
    )
    output_dir = output_root / artifact_id
    write_artifact(
        output_dir,
        artifact_id=artifact_id,
        source_model=model,
        source_revision=revision,
        calib_id=calib_id,
        gptq_config=gptq_payload,
        linears=linears,
    )
    with (output_dir / "gptq_report.jsonl").open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, sort_keys=True) + "\n")
    return output_dir


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/ptq/artifacts"))
    parser.add_argument("--blocks", default="0")
    parser.add_argument("--groups")
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--max-cholesky-retries", type=int, default=3)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli = _parse_args()
    selected_groups = (
        frozenset(value.strip() for value in cli.groups.split(","))
        if cli.groups
        else None
    )
    print(
        quantize_calibration(
            cli.calibration_dir,
            cli.output_root,
            blocks=tuple(int(value) for value in cli.blocks.split(",")),
            groups=selected_groups,
            config=GPTQConfig(
                percdamp=cli.percdamp,
                max_cholesky_retries=cli.max_cholesky_retries,
            ),
        )
    )
