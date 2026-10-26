"""Run and compare Wan 2.2 PTQ accuracy-ladder traces through rung 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .artifact import load_artifact
from .calibrate import (
    CalibrationSettings,
    MODEL_ID,
    MODEL_REVISION,
    assert_no_prompt_overlap,
    build_eager_bf16_args,
    read_prompts,
)
from .mxfp4_ref import dequantize_mxfp4, quantize_mxfp4
from .provenance import runtime_manifest


@lru_cache(maxsize=4)
def _validated_artifact(path: str):
    return load_artifact(path, expected_source_revision=MODEL_REVISION)


def _tensor_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract tensor from {type(output).__name__}")


def _sample_rows(tensor: torch.Tensor, count: int = 16) -> torch.Tensor:
    flat = tensor.reshape(-1, tensor.shape[-1])
    if flat.shape[0] <= count:
        return flat.detach().cpu()
    indices = torch.linspace(
        0, flat.shape[0] - 1, count, device=flat.device
    ).round().long()
    return flat.index_select(0, indices).detach().cpu()


@dataclass
class ErrorSums:
    error2: float = 0.0
    signal2: float = 0.0
    dot: float = 0.0
    reference2: float = 0.0
    candidate2: float = 0.0
    values: int = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        reference = reference.float()
        candidate = candidate.float()
        self.error2 += (reference - candidate).square().sum().item()
        self.signal2 += reference.square().sum().item()
        self.dot += (reference * candidate).sum().item()
        self.reference2 += reference.square().sum().item()
        self.candidate2 += candidate.square().sum().item()
        self.values += reference.numel()

    def report(self) -> dict:
        relative_mse = self.error2 / max(self.signal2, 1.0e-30)
        return {
            "relative_mse": relative_mse,
            "sqnr_db": -10.0 * math.log10(max(relative_mse, 1.0e-30)),
            "cosine_similarity": self.dot
            / max(math.sqrt(self.reference2 * self.candidate2), 1.0e-30),
            "values": self.values,
        }


def _artifact_block_weights(
    artifact_path: Path,
    component: str,
    block_index: int,
    device: str,
) -> dict[str, torch.Tensor]:
    artifact = _validated_artifact(str(artifact_path))
    prefix = f"{component}.blocks.{block_index}."
    weights = {}
    for full_name, entry in artifact.config["layers"].items():
        if not full_name.startswith(prefix):
            continue
        with safe_open(
            artifact.root / entry["shard"], framework="pt", device="cpu"
        ) as handle:
            packed = handle.get_tensor(entry["packed_key"])
            scale = handle.get_tensor(entry["scale_key"])
        weights[full_name[len(component) + 1 :]] = dequantize_mxfp4(
            packed, scale
        ).to(device=device, dtype=torch.bfloat16)
    return weights


class LayerErrorHooks:
    """Evaluate block-0 weights on deterministic held-out BF16 activations."""

    def __init__(self, pipe, artifact_path: Path, rows: int = 32):
        self.handles = []
        self.sums: dict[tuple[str, str, str], ErrorSums] = defaultdict(ErrorSums)
        self.calls = defaultdict(int)
        self.rows = rows
        for component in ("transformer", "transformer_2"):
            transformer = getattr(pipe, component)
            candidate = _artifact_block_weights(
                artifact_path,
                component,
                0,
                str(next(transformer.parameters()).device),
            )
            for name, module in transformer.blocks[0].named_modules():
                full_name = f"blocks.0.{name}"
                if full_name not in candidate:
                    continue
                packed, scales = quantize_mxfp4(module.weight.detach())
                rtn_weight = dequantize_mxfp4(packed, scales).to(
                    dtype=torch.bfloat16
                )
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._hook(
                            component,
                            full_name,
                            module.weight.detach(),
                            rtn_weight,
                            candidate[full_name],
                            module.bias,
                        )
                    )
                )

    def _hook(self, component, name, reference_weight, rtn_weight, ptq_weight, bias):
        def hook(_module, inputs):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if values.shape[0] > self.rows:
                indices = torch.linspace(
                    0, values.shape[0] - 1, self.rows, device=values.device
                ).round().long()
                values = values.index_select(0, indices)
            reference = F.linear(values, reference_weight, bias)
            act_packed, act_scales = quantize_mxfp4(values)
            quantized_values = dequantize_mxfp4(act_packed, act_scales).to(
                dtype=values.dtype
            )
            call = self.calls[(component, name)]
            self.calls[(component, name)] += 1
            bucket = f"calls_{(call // 4) * 4:02d}_{(call // 4) * 4 + 3:02d}"
            for arm, weight in (("rtn", rtn_weight), ("ptq", ptq_weight)):
                output = F.linear(values, weight, bias)
                self.sums[(component, name, f"{bucket}:{arm}:bf16_act")].update(
                    reference, output
                )
                output = F.linear(quantized_values, weight, bias)
                self.sums[(component, name, f"{bucket}:{arm}:mxfp4_act")].update(
                    reference, output
                )

        return hook

    def remove(self) -> list[dict]:
        for handle in self.handles:
            handle.remove()
        reports = []
        for (expert, layer, key), sums in sorted(self.sums.items()):
            bucket, arm, activation = key.split(":")
            reports.append(
                {
                    "expert": expert,
                    "layer": layer,
                    "timestep_bucket": bucket,
                    "arm": arm,
                    "activation": activation,
                    **sums.report(),
                }
            )
        return reports


class TraceHooks:
    def __init__(self, pipe):
        self.predictions = {"transformer": [], "transformer_2": []}
        self.blocks = {"transformer": defaultdict(list), "transformer_2": defaultdict(list)}
        self.handles = []
        for component in ("transformer", "transformer_2"):
            transformer = getattr(pipe, component)
            self.handles.append(
                transformer.register_forward_hook(self._prediction_hook(component))
            )
            for index, block in enumerate(transformer.blocks):
                self.handles.append(
                    block.register_forward_hook(self._block_hook(component, index))
                )

    def _prediction_hook(self, component):
        def hook(_module, _inputs, output):
            self.predictions[component].append(_tensor_output(output).detach().cpu())

        return hook

    def _block_hook(self, component, index):
        def hook(_module, _inputs, output):
            self.blocks[component][index].append(_sample_rows(_tensor_output(output)))

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def _eval_args(arm: str, artifact_path: Path | None):
    settings = CalibrationSettings(prompts_path=Path("unused"))
    args = build_eager_bf16_args(
        settings,
        1,
        gemm_quantization="fp4" if arm != "bf16" else None,
        ptq_artifact_path=(
            str(artifact_path) if arm == "ptq" and artifact_path else None
        ),
    )
    args.output_type = "latent"
    if arm == "ptq":
        if artifact_path is None:
            raise ValueError("the PTQ arm requires --artifact")
    return args


def _callback(latents: list[torch.Tensor], forced: list[torch.Tensor] | None):
    def callback(_pipe, step, _timestep, callback_kwargs):
        current = callback_kwargs["latents"]
        latents.append(current.detach().cpu())
        if forced is not None:
            callback_kwargs["latents"] = forced[step].to(
                device=current.device, dtype=current.dtype
            )
        return callback_kwargs

    return callback


def run_arm(
    *,
    arm: str,
    prompts_path: Path,
    calibration_prompts_path: Path,
    fixed_latent_path: Path,
    output_root: Path,
    artifact_path: Path | None,
    reference_dir: Path | None,
    max_prompts: int,
    prompt_offset: int,
    force_reference_trajectory: bool,
    layer_metrics: bool,
) -> Path:
    from xfuser.core.distributed import init_distributed_environment
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    if not torch.distributed.is_initialized():
        init_distributed_environment()
    if torch.distributed.get_world_size() != 1:
        raise RuntimeError("Phase-0 accuracy traces require one data-parallel rank")

    calibration_prompts = read_prompts(calibration_prompts_path)
    calibration_set = set(calibration_prompts)
    prompts = [
        prompt for prompt in read_prompts(prompts_path)
        if prompt not in calibration_set
    ][prompt_offset : prompt_offset + max_prompts]
    if len(prompts) < max_prompts:
        raise ValueError("not enough held-out prompts after excluding calibration prompts")
    assert_no_prompt_overlap(calibration_prompts, prompts)
    args = _eval_args(arm, artifact_path)
    model = MODEL_REGISTRY[MODEL_ID](args)
    input_args = {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 3.0,
        "prompt": prompts[0],
        "negative_prompt": model.default_input_values.negative_prompt,
        "seed": 42,
    }
    model.initialize(input_args)
    fixed_latent = torch.load(
        fixed_latent_path, map_location="cpu", weights_only=True
    )

    run_payload = {
        "arm": arm,
        "artifact_id": None,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "prompt_offset": prompt_offset,
        "prompt_count": len(prompts),
        "fixed_latent_sha256": hashlib.sha256(fixed_latent_path.read_bytes()).hexdigest(),
        "force_reference_trajectory": force_reference_trajectory,
        "prompts": [],
    }
    if artifact_path:
        artifact = _validated_artifact(str(artifact_path))
        run_payload["artifact_id"] = artifact.artifact_id
    digest = hashlib.sha256(
        json.dumps(run_payload, sort_keys=True).encode()
    ).hexdigest()[:12]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{arm}-{digest}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    layer_hook = (
        LayerErrorHooks(model.pipe, artifact_path)
        if layer_metrics and arm == "bf16" and artifact_path
        else None
    )
    try:
        for local_index, prompt in enumerate(prompts):
            prompt_index = prompt_offset + local_index
            forced = None
            if force_reference_trajectory:
                if reference_dir is None:
                    raise ValueError("--force-reference-trajectory requires --reference-dir")
                reference = torch.load(
                    reference_dir / f"prompt-{prompt_index:03d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                forced = reference["latents"]
            trace = TraceHooks(model.pipe)
            latents = []
            generator = torch.Generator(
                device=model.pipe._execution_device
            ).manual_seed(42 + prompt_index)
            output = model.pipe(
                height=720,
                width=1280,
                num_frames=81,
                num_inference_steps=20,
                prompt=prompt,
                negative_prompt=model.default_input_values.negative_prompt,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                generator=generator,
                latents=fixed_latent.to(model.pipe._execution_device),
                output_type=(
                    "latent" if force_reference_trajectory else "np"
                ),
                callback_on_step_end=_callback(latents, forced),
                callback_on_step_end_tensor_inputs=["latents"],
            )
            trace.remove()
            if not force_reference_trajectory:
                frames = output.frames if hasattr(output, "frames") else output[0]
                np.savez_compressed(
                    output_dir / f"prompt-{prompt_index:03d}-video.npz",
                    frames=np.asarray(frames),
                )
            torch.save(
                {
                    "latents": latents,
                    "predictions": trace.predictions,
                    "blocks": {
                        expert: dict(values)
                        for expert, values in trace.blocks.items()
                    },
                },
                output_dir / f"prompt-{prompt_index:03d}.pt",
            )
            run_payload["prompts"].append(
                {"index": prompt_index, "text": prompt, "seed": 42 + prompt_index}
            )
    finally:
        if layer_hook is not None:
            reports = layer_hook.remove()
            (output_dir / "rung1-layer-error.json").write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {**run_payload, "runtime": runtime_manifest()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def _compare_lists(reference, candidate) -> dict:
    sums = ErrorSums()
    if len(reference) != len(candidate):
        raise ValueError(f"trace length mismatch: {len(reference)} != {len(candidate)}")
    for expected, actual in zip(reference, candidate):
        sums.update(expected, actual)
    return sums.report()


def _velocity_metrics(reference, candidate, guidance: float) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("velocity trace length mismatch")
    if not reference:
        return {"branches": [], "cfg": [], "overall": ErrorSums().report()}
    steps = []
    if reference[0].shape[0] >= 2:
        steps = [
            ((expected[0], expected[1]), (actual[0], actual[1]))
            for expected, actual in zip(reference, candidate)
        ]
    else:
        if len(reference) % 2:
            raise ValueError("separate-CFG velocity trace has an odd call count")
        for index in range(0, len(reference), 2):
            # WanPipeline calls the conditional branch first, then the
            # unconditional branch when CFG is not parallelized.
            steps.append(
                (
                    (reference[index + 1], reference[index]),
                    (candidate[index + 1], candidate[index]),
                )
            )
    selected = sorted(
        set(
            torch.linspace(0, len(steps) - 1, min(5, len(steps)))
            .round()
            .int()
            .tolist()
        )
    )
    reports = []
    overall = ErrorSums()
    for step in selected:
        (ref_uncond, ref_cond), (cand_uncond, cand_cond) = steps[step]
        uncond = ErrorSums()
        cond = ErrorSums()
        cfg = ErrorSums()
        uncond.update(ref_uncond, cand_uncond)
        cond.update(ref_cond, cand_cond)
        ref_cfg = ref_uncond + guidance * (ref_cond - ref_uncond)
        cand_cfg = cand_uncond + guidance * (cand_cond - cand_uncond)
        cfg.update(ref_cfg, cand_cfg)
        overall.update(ref_cfg, cand_cfg)
        reports.append(
            {
                "step_within_expert": step,
                "unconditional": uncond.report(),
                "conditional": cond.report(),
                "cfg_combined": cfg.report(),
            }
        )
    return {"sampled_steps": reports, "cfg_combined": overall.report()}


def _ssim_frames(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    # Standard local SSIM with an 11x11 uniform window; inputs are NCHW in [0, 1].
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(reference, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(candidate, 11, stride=1, padding=5)
    var_x = F.avg_pool2d(reference.square(), 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(candidate.square(), 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(reference * candidate, 11, 1, 5) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return value.mean().item()


def compare_runs(reference_dir: Path, arm_dirs: list[Path], output: Path) -> dict:
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    result = {"reference": str(reference_dir), "arms": {}}
    layer_report_path = reference_dir / "rung1-layer-error.json"
    if layer_report_path.exists():
        layer_rows = json.loads(layer_report_path.read_text(encoding="utf-8"))
        by_key = defaultdict(dict)
        for row in layer_rows:
            key = (
                row["expert"],
                row["layer"],
                row["timestep_bucket"],
                row["activation"],
            )
            by_key[key][row["arm"]] = row
        result["rung1"] = []
        for key, arms in sorted(by_key.items()):
            if "rtn" not in arms or "ptq" not in arms:
                continue
            rtn_error = arms["rtn"]["relative_mse"]
            ptq_error = arms["ptq"]["relative_mse"]
            result["rung1"].append(
                {
                    "expert": key[0],
                    "layer": key[1],
                    "timestep_bucket": key[2],
                    "activation": key[3],
                    "rtn": arms["rtn"],
                    "ptq": arms["ptq"],
                    "gap_closed": (rtn_error - ptq_error) / max(rtn_error, 1.0e-30),
                }
            )
    grouped = defaultdict(dict)
    for arm_dir in arm_dirs:
        manifest = json.loads((arm_dir / "manifest.json").read_text(encoding="utf-8"))
        mode = "forced" if manifest["force_reference_trajectory"] else "free"
        grouped[manifest["arm"]][mode] = (arm_dir, manifest)
    for arm, modes in grouped.items():
        if "forced" not in modes or "free" not in modes:
            raise ValueError(f"{arm} requires both forced and free trajectory runs")
        forced_dir, forced_manifest = modes["forced"]
        free_dir, free_manifest = modes["free"]
        prompts = []
        for prompt in reference_manifest["prompts"]:
            index = prompt["index"]
            reference = torch.load(
                reference_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            candidate = torch.load(
                forced_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            free_candidate = torch.load(
                free_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            block_metrics = {}
            velocity_metrics = {}
            for expert in ("transformer", "transformer_2"):
                velocity_metrics[expert] = _velocity_metrics(
                    reference["predictions"][expert],
                    candidate["predictions"][expert],
                    4.0 if expert == "transformer" else 3.0,
                )
                for block_index, expected in reference["blocks"][expert].items():
                    block_metrics[f"{expert}.blocks.{block_index}"] = _compare_lists(
                        expected,
                        candidate["blocks"][expert][block_index],
                    )
            latent_metrics = [
                _compare_lists([expected], [actual])
                for expected, actual in zip(reference["latents"], free_candidate["latents"])
            ]
            reference_video = np.load(
                reference_dir / f"prompt-{index:03d}-video.npz"
            )["frames"]
            candidate_video = np.load(
                free_dir / f"prompt-{index:03d}-video.npz"
            )["frames"]
            ref_tensor = torch.from_numpy(reference_video).float()
            cand_tensor = torch.from_numpy(candidate_video).float()
            if ref_tensor.max() > 1:
                ref_tensor /= 255
                cand_tensor /= 255
            images_ref = ref_tensor.reshape(-1, *ref_tensor.shape[-3:]).permute(0, 3, 1, 2)
            images_cand = cand_tensor.reshape(-1, *cand_tensor.shape[-3:]).permute(0, 3, 1, 2)
            mse = (images_ref - images_cand).square().flatten(1).mean(1)
            prompts.append(
                {
                    "index": index,
                    "rung2": {
                        "block0_bf16_input": {
                            key: value
                            for key, value in block_metrics.items()
                            if key.endswith(".blocks.0")
                        },
                        "propagated_blocks": block_metrics,
                    },
                    "rung3": {"velocity": velocity_metrics},
                    "rung4": {
                        "latent_drift": latent_metrics,
                        "frame_psnr_db": (
                            -10 * torch.log10(mse.clamp_min(1.0e-30))
                        ).tolist(),
                        "frame_ssim": _ssim_frames(images_ref, images_cand),
                        "lpips": None,
                        "clip_text_video_similarity": None,
                    },
                }
            )
        result["arms"][arm] = {
            "forced_run": str(forced_dir),
            "free_run": str(free_dir),
            "artifact_id": free_manifest.get("artifact_id"),
            "prompts": prompts,
        }
    if "rtn" in result["arms"] and "ptq" in result["arms"]:
        gaps = []
        for rtn_prompt, ptq_prompt in zip(
            result["arms"]["rtn"]["prompts"],
            result["arms"]["ptq"]["prompts"],
        ):
            entry = {
                "index": rtn_prompt["index"],
                "rung2": {},
                "rung3": {},
                "rung4": [],
            }
            for block, rtn_metric in rtn_prompt["rung2"]["propagated_blocks"].items():
                ptq_metric = ptq_prompt["rung2"]["propagated_blocks"][block]
                entry["rung2"][block] = (
                    rtn_metric["relative_mse"] - ptq_metric["relative_mse"]
                ) / max(rtn_metric["relative_mse"], 1.0e-30)
            for expert, rtn_metric in rtn_prompt["rung3"]["velocity"].items():
                ptq_metric = ptq_prompt["rung3"]["velocity"][expert]
                entry["rung3"][expert] = (
                    rtn_metric["cfg_combined"]["relative_mse"]
                    - ptq_metric["cfg_combined"]["relative_mse"]
                ) / max(rtn_metric["cfg_combined"]["relative_mse"], 1.0e-30)
            for rtn_metric, ptq_metric in zip(
                rtn_prompt["rung4"]["latent_drift"],
                ptq_prompt["rung4"]["latent_drift"],
            ):
                entry["rung4"].append(
                    (rtn_metric["relative_mse"] - ptq_metric["relative_mse"])
                    / max(rtn_metric["relative_mse"], 1.0e-30)
                )
            gaps.append(entry)
        result["gap_closed"] = gaps
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def aggregate_reports(report_paths: list[Path], output: Path) -> dict:
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    gaps = [report["gap_closed"][0] for report in reports]

    def mean_at(path):
        values = []
        for item in gaps:
            value = item
            for key in path:
                value = value[key]
            values.append(value)
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "values": values,
        }

    summary = {
        "reports": [str(path) for path in report_paths],
        "prompt_count": len(reports),
        "gap_closed": {
            "rung2": {
                name: mean_at(("rung2", name))
                for name in (
                    "transformer.blocks.0",
                    "transformer.blocks.39",
                    "transformer_2.blocks.0",
                    "transformer_2.blocks.39",
                )
            },
            "rung3": {
                expert: mean_at(("rung3", expert))
                for expert in ("transformer", "transformer_2")
            },
            "rung4_by_step": [
                {
                    "mean": statistics.fmean(
                        item["rung4"][step] for item in gaps
                    ),
                    "median": statistics.median(
                        item["rung4"][step] for item in gaps
                    ),
                }
                for step in range(len(gaps[0]["rung4"]))
            ],
        },
        "arms": {},
    }
    for arm in ("rtn", "ptq"):
        prompts = [report["arms"][arm]["prompts"][0] for report in reports]
        ssim = [prompt["rung4"]["frame_ssim"] for prompt in prompts]
        psnr = [
            statistics.fmean(prompt["rung4"]["frame_psnr_db"])
            for prompt in prompts
        ]
        final_latent = [
            prompt["rung4"]["latent_drift"][-1]["relative_mse"]
            for prompt in prompts
        ]
        summary["arms"][arm] = {
            "frame_ssim": {
                "mean": statistics.fmean(ssim),
                "median": statistics.median(ssim),
                "values": ssim,
            },
            "frame_psnr_db": {
                "mean": statistics.fmean(psnr),
                "median": statistics.median(psnr),
                "values": psnr,
            },
            "final_latent_relative_mse": {
                "mean": statistics.fmean(final_latent),
                "median": statistics.median(final_latent),
                "values": final_latent,
            },
        }
    rung1 = reports[0].get("rung1")
    if rung1:
        summary["rung1"] = {}
        for activation in ("bf16_act", "mxfp4_act"):
            values = [
                row["gap_closed"]
                for row in rung1
                if row["activation"] == activation
            ]
            summary["rung1"][activation] = {
                "mean_gap_closed": statistics.fmean(values),
                "median_gap_closed": statistics.median(values),
                "measurements": len(values),
            }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--arm", choices=("bf16", "rtn", "ptq"), required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--calibration-prompts", type=Path, required=True)
    run.add_argument("--fixed-latent", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=Path("/data/ptq/eval"))
    run.add_argument("--artifact", type=Path)
    run.add_argument("--reference-dir", type=Path)
    run.add_argument("--max-prompts", type=int, default=1)
    run.add_argument("--prompt-offset", type=int, default=0)
    run.add_argument("--force-reference-trajectory", action="store_true")
    run.add_argument("--layer-metrics", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference-dir", type=Path, required=True)
    compare.add_argument("--arm-dir", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    if cli.command == "run":
        print(
            run_arm(
                arm=cli.arm,
                prompts_path=cli.prompts,
                calibration_prompts_path=cli.calibration_prompts,
                fixed_latent_path=cli.fixed_latent,
                output_root=cli.output_root,
                artifact_path=cli.artifact,
                reference_dir=cli.reference_dir,
                max_prompts=cli.max_prompts,
                prompt_offset=cli.prompt_offset,
                force_reference_trajectory=cli.force_reference_trajectory,
                layer_metrics=cli.layer_metrics,
            )
        )
    else:
        print(compare_runs(cli.reference_dir, cli.arm_dir, cli.output)["arms"].keys())
