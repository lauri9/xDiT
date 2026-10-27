"""Fit, serialize, and apply Wan self-attention PTQ transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ATTN_FORMAT_VERSION = 1
ATTN_TENSOR_FILE = "attention_transforms.safetensors"
ATTN_TENSOR_ROLES = frozenset(
    {"k_shift", "v_bias_delta", "out_bias_delta", "scale"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: Mapping) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _moment_mean(stats: Mapping) -> torch.Tensor:
    count = int(stats["token_count"])
    if count <= 0:
        raise ValueError("attention statistics have no observed tokens")
    total = torch.as_tensor(stats["sum"], dtype=torch.float64, device="cpu")
    mean = (total / count).to(torch.float32)
    if not torch.isfinite(mean).all():
        raise ValueError("attention statistics produced a non-finite mean")
    return mean


def fit_t1(key_stats: Mapping) -> torch.Tensor:
    """Fit the post-RoPE per-head K shift."""

    return _moment_mean(key_stats)


def fit_t2(value_stats: Mapping) -> torch.Tensor:
    """Fit the per-head V shift."""

    return _moment_mean(value_stats)


def fit_t3(
    query_stats: Mapping,
    key_stats: Mapping,
    *,
    alpha: float = 0.5,
    power_of_two: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fit pair-tied SmoothQuant scales for ``Q / s`` and ``K * s``."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("T3 alpha must be in [0, 1]")
    if eps <= 0:
        raise ValueError("T3 epsilon must be positive")
    query = torch.as_tensor(query_stats["absmax"], dtype=torch.float32, device="cpu")
    key = torch.as_tensor(key_stats["absmax"], dtype=torch.float32, device="cpu")
    if query.shape != key.shape or query.ndim != 2:
        raise ValueError("T3 Q/K absmax tensors must have the same [H, D] shape")
    if query.shape[-1] % 2:
        raise ValueError("T3 requires an even head dimension for RoPE pairs")
    # One scale per RoPE pair is required for diag(s) to commute with RoPE.
    query_pair = query.unflatten(-1, (-1, 2)).amax(dim=-1)
    key_pair = key.unflatten(-1, (-1, 2)).amax(dim=-1)
    both_zero = (query_pair == 0) & (key_pair == 0)
    scale_pair = query_pair.clamp_min(eps).pow(alpha) / key_pair.clamp_min(
        eps
    ).pow(1.0 - alpha)
    scale_pair[both_zero] = 1.0
    if power_of_two:
        scale_pair = torch.pow(2.0, torch.round(torch.log2(scale_pair)))
    scale = scale_pair.repeat_interleave(2, dim=-1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("T3 fitting produced an invalid scale")
    return scale


@dataclass(frozen=True)
class AttentionTransform:
    k_shift: torch.Tensor
    v_shift: torch.Tensor
    scale: torch.Tensor


def fit_attention_transform(
    block_stats: Mapping,
    *,
    enable_t1: bool = True,
    enable_t2: bool = True,
    enable_t3: bool = True,
    alpha: float = 0.5,
    power_of_two: bool = False,
) -> AttentionTransform:
    shape = torch.as_tensor(block_stats["k"]["sum"]).shape
    ones = torch.ones(shape, dtype=torch.float32)
    zeros = torch.zeros(shape, dtype=torch.float32)
    return AttentionTransform(
        k_shift=fit_t1(block_stats["k"]) if enable_t1 else zeros.clone(),
        v_shift=fit_t2(block_stats["v"]) if enable_t2 else zeros.clone(),
        scale=(
            fit_t3(
                block_stats["q"],
                block_stats["k"],
                alpha=alpha,
                power_of_two=power_of_two,
            )
            if enable_t3
            else ones
        ),
    )


def fold_attention_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str,
    transform: AttentionTransform,
) -> dict[str, torch.Tensor]:
    """Return a state dict with exact-real T2/T3 folds applied."""

    result = dict(state_dict)
    flat_v = transform.v_shift.flatten()
    flat_s = transform.scale.flatten()
    names = {
        "v_bias": f"{prefix}.to_v.bias",
        "out_weight": f"{prefix}.to_out.0.weight",
        "out_bias": f"{prefix}.to_out.0.bias",
        "q_norm": f"{prefix}.norm_q.weight",
        "k_norm": f"{prefix}.norm_k.weight",
    }
    missing = [name for name in names.values() if name not in state_dict]
    if missing:
        raise KeyError(f"attention fold is missing tensors: {missing}")
    result[names["v_bias"]] = state_dict[names["v_bias"]] - flat_v.to(
        state_dict[names["v_bias"]]
    )
    output_delta = torch.mv(
        state_dict[names["out_weight"]].float(), flat_v.float()
    )
    result[names["out_bias"]] = state_dict[names["out_bias"]] + output_delta.to(
        state_dict[names["out_bias"]]
    )
    result[names["q_norm"]] = state_dict[names["q_norm"]] / flat_s.to(
        state_dict[names["q_norm"]]
    )
    result[names["k_norm"]] = state_dict[names["k_norm"]] * flat_s.to(
        state_dict[names["k_norm"]]
    )
    return result


@dataclass(frozen=True)
class AttentionArtifact:
    root: Path
    config: dict
    tensors: Mapping[str, torch.Tensor]

    @property
    def artifact_id(self) -> str:
        return str(self.config["artifact_id"])


def write_attention_artifact(
    root: str | Path,
    *,
    artifact_id: str,
    source_model: str,
    source_revision: str,
    calib_id: str,
    fit_config: Mapping,
    tensors: Mapping[str, torch.Tensor],
    blocks: Mapping[str, Mapping],
) -> AttentionArtifact:
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"artifact directory already exists: {root}")
    if not tensors or not blocks:
        raise ValueError("attention artifact must contain transforms")
    cpu_tensors = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, tensor in tensors.items()
    }
    if any(not torch.isfinite(tensor).all() for tensor in cpu_tensors.values()):
        raise ValueError("attention artifact tensors must be finite")
    root.mkdir(parents=True)
    tensor_path = root / ATTN_TENSOR_FILE
    save_file(cpu_tensors, tensor_path)
    config = {
        "artifact_type": "attention_transforms",
        "format_version": ATTN_FORMAT_VERSION,
        "artifact_id": artifact_id,
        "source": {"model": source_model, "revision": source_revision},
        "calib_id": calib_id,
        "fit_config": dict(fit_config),
        "fit_config_hash": _config_hash(fit_config),
        "tensor_file": ATTN_TENSOR_FILE,
        "tensor_sha256": _sha256_file(tensor_path),
        "blocks": dict(blocks),
    }
    config["metadata_sha256"] = _config_hash(config)
    (root / "attn_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return AttentionArtifact(root, config, cpu_tensors)


def load_attention_artifact(
    root: str | Path, *, expected_source_revision: str
) -> AttentionArtifact:
    root = Path(root)
    config = json.loads((root / "attn_config.json").read_text(encoding="utf-8"))
    if config.get("artifact_type") != "attention_transforms":
        raise ValueError("not an attention-transform artifact")
    if config.get("format_version") != ATTN_FORMAT_VERSION:
        raise ValueError("unsupported attention artifact format version")
    if config.get("source", {}).get("revision") != expected_source_revision:
        raise ValueError("attention artifact BF16 source revision mismatch")
    metadata = dict(config)
    metadata_hash = metadata.pop("metadata_sha256", None)
    if metadata_hash != _config_hash(metadata):
        raise ValueError("attention artifact metadata hash mismatch")
    if config.get("fit_config_hash") != _config_hash(config.get("fit_config", {})):
        raise ValueError("attention artifact fit-config hash mismatch")
    tensor_path = root / config.get("tensor_file", "")
    if _sha256_file(tensor_path) != config.get("tensor_sha256"):
        raise ValueError("attention artifact tensor hash mismatch")
    tensors = load_file(tensor_path, device="cpu")
    expected_keys: set[str] = set()
    for name, entry in config.get("blocks", {}).items():
        if entry.get("shape") is None:
            raise ValueError(f"attention artifact has no shape for {name}")
        if set(entry.get("tensors", {})) != ATTN_TENSOR_ROLES:
            raise ValueError(f"attention artifact tensor roles mismatch for {name}")
        expected_keys.update(entry.get("tensors", {}).values())
    if set(tensors) != expected_keys:
        raise ValueError("attention artifact tensor set mismatch")
    for name, entry in config["blocks"].items():
        shape = tuple(entry["shape"])
        for role, key in entry["tensors"].items():
            expected_shape = (shape[0] * shape[1],) if role == "out_bias_delta" else shape
            if tuple(tensors[key].shape) != expected_shape:
                raise ValueError(
                    f"attention artifact {role} shape mismatch for {name}"
                )
    return AttentionArtifact(root, config, tensors)


def apply_attention_artifact(
    components: Mapping[str, torch.nn.Module], artifact: AttentionArtifact
) -> None:
    """Apply T1 buffers and T2/T3 bias/norm deltas in place."""

    targets = []
    for full_name, entry in artifact.config["blocks"].items():
        component_name, separator, module_name = full_name.partition(".")
        if not separator or component_name not in components:
            raise ValueError(f"attention artifact has unknown component {full_name}")
        attention = components[component_name].get_submodule(module_name)
        if getattr(attention, "_xfuser_attention_artifact_id", None) is not None:
            raise RuntimeError(f"attention transform already applied to {full_name}")
        shape = tuple(entry["shape"])
        if tuple(attention.k_shift.shape) != shape:
            raise ValueError(f"attention artifact shape mismatch for {full_name}")
        flat_size = shape[0] * shape[1]
        parameters = (
            attention.to_v.bias,
            attention.to_out[0].bias,
            attention.norm_q.weight,
            attention.norm_k.weight,
        )
        if any(value is None or value.numel() != flat_size for value in parameters):
            raise ValueError(f"attention fold parameter mismatch for {full_name}")
        values = {
            role: artifact.tensors[key]
            for role, key in entry["tensors"].items()
        }
        targets.append((full_name, attention, values))

    with torch.no_grad():
        for full_name, attention, values in targets:
            flat_scale = values["scale"].flatten()
            attention.k_shift.copy_(values["k_shift"].to(attention.k_shift))
            attention.to_v.bias.add_(
                values["v_bias_delta"].flatten().to(attention.to_v.bias)
            )
            attention.to_out[0].bias.add_(
                values["out_bias_delta"].to(attention.to_out[0].bias)
            )
            attention.norm_q.weight.div_(flat_scale.to(attention.norm_q.weight))
            attention.norm_k.weight.mul_(flat_scale.to(attention.norm_k.weight))
            attention._xfuser_use_k_shift = bool(
                artifact.config["fit_config"].get("t1", True)
            )
            attention._xfuser_attention_artifact_id = artifact.artifact_id


def _checkpoint_map(model: str, revision: str, component: str) -> dict[str, str]:
    index = hf_hub_download(
        repo_id=model,
        filename=f"{component}/diffusion_pytorch_model.safetensors.index.json",
        revision=revision,
        local_files_only=True,
    )
    raw_map = json.loads(Path(index).read_text(encoding="utf-8"))["weight_map"]
    paths: dict[str, str] = {}
    result: dict[str, str] = {}
    for key, filename in raw_map.items():
        if filename not in paths:
            paths[filename] = hf_hub_download(
                repo_id=model,
                filename=f"{component}/{filename}",
                revision=revision,
                local_files_only=True,
            )
        result[key] = paths[filename]
    return result


def _read_checkpoint_tensor(weight_map: Mapping[str, str], key: str) -> torch.Tensor:
    if key not in weight_map:
        raise KeyError(f"checkpoint has no tensor {key}")
    with safe_open(weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def fit_calibration_attention(
    calibration_dir: Path,
    output_root: Path,
    *,
    enable_t1: bool = True,
    enable_t2: bool = True,
    enable_t3: bool = True,
    alpha: float = 0.5,
    power_of_two: bool = False,
) -> Path:
    record = json.loads(
        (calibration_dir / "calib_record.json").read_text(encoding="utf-8")
    )
    settings = record["settings"]
    attention_record = record.get("attention_stats")
    if not attention_record or not attention_record.get("enabled"):
        raise ValueError("calibration record contains no attention statistics")
    fit_config = {
        "t1": enable_t1,
        "t2": enable_t2,
        "t3": enable_t3,
        "alpha": alpha,
        "power_of_two": power_of_two,
        "observer_placement": "post_rope_attention_input_pre_usp",
    }
    tensors: dict[str, torch.Tensor] = {}
    blocks: dict[str, dict] = {}
    for component, filename in attention_record["files"].items():
        saved = torch.load(
            calibration_dir / filename,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if saved.get("format_version") != 1:
            raise ValueError("unsupported attention statistics format")
        weight_map = _checkpoint_map(
            settings["model"], settings["model_revision"], component
        )
        for module_name, block_stats in saved["groups"].items():
            full_name = f"{component}.{module_name}"
            transform = fit_attention_transform(
                block_stats,
                enable_t1=enable_t1,
                enable_t2=enable_t2,
                enable_t3=enable_t3,
                alpha=alpha,
                power_of_two=power_of_two,
            )
            prefix = full_name
            keys = {
                "k_shift": f"{prefix}.k_shift",
                "v_bias_delta": f"{prefix}.v_bias_delta",
                "out_bias_delta": f"{prefix}.out_bias_delta",
                "scale": f"{prefix}.scale",
            }
            flat_v = transform.v_shift.flatten()
            out_weight = _read_checkpoint_tensor(
                weight_map, f"{module_name}.to_out.0.weight"
            )
            tensors[keys["k_shift"]] = transform.k_shift
            tensors[keys["v_bias_delta"]] = -transform.v_shift
            tensors[keys["out_bias_delta"]] = torch.mv(
                out_weight.float(), flat_v.float()
            )
            tensors[keys["scale"]] = transform.scale
            blocks[full_name] = {
                "shape": list(transform.k_shift.shape),
                "tensors": keys,
            }
    digest = _config_hash(
        {
            "calib_id": record["calib_id"],
            "source_revision": settings["model_revision"],
            **fit_config,
        }
    )[:12]
    artifact_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{digest}"
    )
    output_dir = output_root / artifact_id
    write_attention_artifact(
        output_dir,
        artifact_id=artifact_id,
        source_model=settings["model"],
        source_revision=settings["model_revision"],
        calib_id=record["calib_id"],
        fit_config=fit_config,
        tensors=tensors,
        blocks=blocks,
    )
    return output_dir


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("/data/ptq/attention-artifacts")
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--power-of-two", action="store_true")
    parser.add_argument("--no-t1", action="store_true")
    parser.add_argument("--no-t2", action="store_true")
    parser.add_argument("--no-t3", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    print(
        fit_calibration_attention(
            cli.calibration_dir,
            cli.output_root,
            enable_t1=not cli.no_t1,
            enable_t2=not cli.no_t2,
            enable_t3=not cli.no_t3,
            alpha=cli.alpha,
            power_of_two=cli.power_of_two,
        )
    )
