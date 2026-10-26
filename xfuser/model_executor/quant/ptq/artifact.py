"""Versioned, hashed, write-once MXFP4 PTQ artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .prequantized_load import PackedLinearSpec


FORMAT_VERSION = 1
PACK_LAYOUT_VERSION = 1
PACK_LAYOUT = "e2m1-low-nibble-first-unshuffled"
SCALE_LAYOUT = "e8m0-per-1x32-unshuffled"


@dataclass(frozen=True)
class QuantizedLinear:
    packed: torch.Tensor
    scale: torch.Tensor


@dataclass(frozen=True)
class PTQArtifact:
    root: Path
    config: dict

    @property
    def artifact_id(self) -> str:
        return str(self.config["artifact_id"])

    def specs_for_component(self, component_name: str) -> tuple[PackedLinearSpec, ...]:
        prefix = f"{component_name}."
        specs = []
        for full_name, entry in self.config["layers"].items():
            if not full_name.startswith(prefix):
                continue
            module_name = full_name[len(prefix) :]
            out_features, in_features = entry["weight_shape"]
            specs.append(
                PackedLinearSpec(
                    module_name=module_name,
                    packed_key=entry["packed_key"],
                    scale_key=entry["scale_key"],
                    packed_shape=tuple(entry["packed_shape"]),
                    shuffled_scale_shape=(
                        (out_features + 255) // 256 * 256,
                        ((in_features // 32) + 7) // 8 * 8,
                    ),
                    artifact_path=str(self.root / entry["shard"]),
                )
            )
        return tuple(specs)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Mapping) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_pair(name: str, pair: QuantizedLinear) -> tuple[int, int]:
    if pair.packed.device.type != "cpu" or pair.scale.device.type != "cpu":
        raise ValueError(f"{name}: artifact tensors must be on CPU")
    packed = pair.packed.view(torch.uint8)
    scale = pair.scale.view(torch.uint8)
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError(f"{name}: packed weight and scale must be two-dimensional")
    out_features, packed_k = packed.shape
    in_features = packed_k * 2
    if in_features % 32 or tuple(scale.shape) != (out_features, in_features // 32):
        raise ValueError(f"{name}: invalid unshuffled per-1x32 MXFP4 shapes")
    return out_features, in_features


def write_artifact(
    root: str | Path,
    *,
    artifact_id: str,
    source_model: str,
    source_revision: str,
    calib_id: str,
    gptq_config: Mapping,
    linears: Mapping[str, QuantizedLinear],
    max_shard_bytes: int = 4 * 1024**3,
) -> PTQArtifact:
    """Write a bounded-shard artifact; an existing directory is never reused."""

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"artifact directory already exists: {root}")
    if not linears:
        raise ValueError("an artifact must contain at least one quantized linear")
    root.mkdir(parents=True)

    groups: list[list[tuple[str, QuantizedLinear, int, int]]] = []
    current: list[tuple[str, QuantizedLinear, int, int]] = []
    current_bytes = 0
    for name, pair in sorted(linears.items()):
        out_features, in_features = _validate_pair(name, pair)
        size = pair.packed.numel() + pair.scale.numel()
        if current and current_bytes + size > max_shard_bytes:
            groups.append(current)
            current = []
            current_bytes = 0
        current.append((name, pair, out_features, in_features))
        current_bytes += size
    groups.append(current)

    layers = {}
    shard_hashes = {}
    total = len(groups)
    for index, group in enumerate(groups, start=1):
        filename = f"weights-{index:05d}-of-{total:05d}.safetensors"
        tensors = {}
        for name, pair, out_features, in_features in group:
            packed_key = f"{name}.weight_packed"
            scale_key = f"{name}.weight_scale"
            packed = pair.packed.view(torch.uint8).contiguous()
            scale = pair.scale.view(torch.uint8).contiguous()
            tensors[packed_key] = packed
            tensors[scale_key] = scale
            layers[name] = {
                "weight_shape": [out_features, in_features],
                "packed_shape": list(packed.shape),
                "scale_shape": list(scale.shape),
                "packed_key": packed_key,
                "scale_key": scale_key,
                "shard": filename,
            }
        shard_path = root / filename
        save_file(tensors, shard_path)
        shard_hashes[filename] = _sha256_file(shard_path)

    config = {
        "format_version": FORMAT_VERSION,
        "pack_layout_version": PACK_LAYOUT_VERSION,
        "pack_layout": PACK_LAYOUT,
        "scale_layout": SCALE_LAYOUT,
        "artifact_id": artifact_id,
        "source": {"model": source_model, "revision": source_revision},
        "calib_id": calib_id,
        "gptq_config": dict(gptq_config),
        "gptq_config_hash": config_hash(gptq_config),
        "layers": layers,
        "shards": shard_hashes,
    }
    (root / "quant_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PTQArtifact(root=root, config=config)


def load_artifact(
    root: str | Path,
    *,
    expected_source_revision: str,
    expected_layers: Mapping[str, tuple[int, int]] | None = None,
) -> PTQArtifact:
    """Validate all metadata and shard hashes before returning an artifact."""

    root = Path(root)
    config = json.loads((root / "quant_config.json").read_text(encoding="utf-8"))
    if config.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported PTQ artifact format version")
    if config.get("pack_layout_version") != PACK_LAYOUT_VERSION:
        raise ValueError("unsupported PTQ artifact pack-layout version")
    if config.get("pack_layout") != PACK_LAYOUT or config.get("scale_layout") != SCALE_LAYOUT:
        raise ValueError("unsupported PTQ artifact tensor layout")
    if config.get("source", {}).get("revision") != expected_source_revision:
        raise ValueError("PTQ artifact BF16 source revision mismatch")

    actual_layers = {
        name: tuple(entry["weight_shape"])
        for name, entry in config.get("layers", {}).items()
    }
    if expected_layers is not None and actual_layers != dict(expected_layers):
        raise ValueError("PTQ artifact layer set or shape mismatch")

    for filename, expected_hash in config.get("shards", {}).items():
        path = root / filename
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"PTQ artifact shard hash mismatch: {filename}")

    by_shard: dict[str, list[tuple[str, dict]]] = {}
    for name, entry in config.get("layers", {}).items():
        by_shard.setdefault(entry["shard"], []).append((name, entry))
    for filename, entries in by_shard.items():
        with safe_open(root / filename, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name, entry in entries:
                if entry["packed_key"] not in keys or entry["scale_key"] not in keys:
                    raise ValueError(f"PTQ artifact tensors missing for {name}")
                packed = handle.get_slice(entry["packed_key"])
                scale = handle.get_slice(entry["scale_key"])
                if list(packed.get_shape()) != entry["packed_shape"]:
                    raise ValueError(f"PTQ artifact packed shape mismatch for {name}")
                if list(scale.get_shape()) != entry["scale_shape"]:
                    raise ValueError(f"PTQ artifact scale shape mismatch for {name}")

    return PTQArtifact(root=root, config=config)
