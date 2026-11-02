"""Promote selected Wan block GEMM groups to a higher-precision MX format.

Production RTN packs MXFP4 weights at load. This replaces the chosen linears
of one transformer with ``xFuserA6W4Linear`` (MXFP6 activations, MXFP4
weights) built from the same BF16 checkpoint tensors, before the first
forward and before any ``torch.compile``. The default path is untouched.
"""
from __future__ import annotations

from typing import Callable

import torch

GROUP_LINEARS = {
    "self_qkv": ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    "self_out": ("attn1.to_out.0",),
    "cross_q": ("attn2.to_q",),
    "cross_kv": ("attn2.to_k", "attn2.to_v"),
    "cross_out": ("attn2.to_out.0",),
    "ffn_in": ("ffn.net.0.proj",),
    "ffn_out": ("ffn.net.2",),
}


def _default_factory(fmt: str):
    if fmt != "a6w4":
        raise ValueError(f"unsupported selective-precision format {fmt!r}")
    from xfuser.model_executor.layers.mixed_mxfp_linear import xFuserA6W4Linear

    return xFuserA6W4Linear


def checkpoint_weight_source(component: str, device="cuda") -> Callable[[str], torch.Tensor]:
    """``key -> BF16 tensor`` reader over the pinned Wan checkpoint component."""
    from .calibrate import MODEL_ID, MODEL_REVISION
    from .quantize import _checkpoint_map, _read_weight

    weight_map = _checkpoint_map(MODEL_ID, MODEL_REVISION, component)
    return lambda key: _read_weight(weight_map, key, device).to(torch.bfloat16)


def artifact_weight_source(artifact_root, component: str, device="cuda") -> Callable[[str], torch.Tensor]:
    """``key -> BF16`` reader taking ``.weight`` tensors from a GPTQ artifact.

    Weights are the artifact's dequantized MXFP4 values (exactly on the MXFP4
    grid and exactly representable in BF16); biases and anything the artifact
    does not cover come from the pinned checkpoint. The artifact is fully
    validated (revision, shard hashes, shapes) before use.
    """
    from safetensors import safe_open

    from .artifact import load_artifact
    from .calibrate import MODEL_REVISION
    from .mxfp4_ref import dequantize_mxfp4

    artifact = load_artifact(artifact_root, expected_source_revision=MODEL_REVISION)
    layers = artifact.config["layers"]
    checkpoint = checkpoint_weight_source(component, device)

    def read(key: str) -> torch.Tensor:
        name, _, kind = key.rpartition(".")
        entry = layers.get(f"{component}.{name}")
        if kind != "weight" or entry is None:
            if kind == "weight":
                raise KeyError(f"artifact {artifact.artifact_id} does not cover {component}.{name}")
            return checkpoint(key)
        with safe_open(artifact.root / entry["shard"], framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(entry["packed_key"])
            scale = handle.get_tensor(entry["scale_key"])
        values = dequantize_mxfp4(packed.to(device), scale.to(device))
        if tuple(values.shape) != tuple(entry["weight_shape"]):
            raise ValueError(f"artifact weight shape mismatch for {name}")
        weight = values.to(torch.bfloat16)
        if not torch.equal(weight.float(), values):
            raise ValueError(f"artifact weight for {name} is not exactly representable in BF16")
        return weight

    read.artifact_id = artifact.artifact_id
    return read


def _replace(root: torch.nn.Module, name: str, module: torch.nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if isinstance(parent, (torch.nn.ModuleList, torch.nn.Sequential)) and child.isdigit():
        parent[int(child)] = module
    else:
        setattr(parent, child, module)


@torch.no_grad()
def install_selective_precision(
    transformer,
    groups,
    *,
    weight_source: Callable[[str], torch.Tensor],
    fmt: str = "a6w4",
    blocks=None,
    factory=None,
) -> list[str]:
    """Replace ``groups`` linears in ``blocks`` (default all) of ``transformer``.

    ``weight_source(key)`` returns the BF16 checkpoint tensor for a block-local
    key such as ``blocks.3.ffn.net.2.weight`` (and ``.bias``). Returns the
    replaced module names.
    """
    unknown = set(groups) - set(GROUP_LINEARS)
    if unknown or not groups:
        raise ValueError(f"unknown or empty GEMM groups: {sorted(unknown) or groups}")
    factory = factory or _default_factory(fmt)
    indices = range(len(transformer.blocks)) if blocks is None else blocks
    replaced = []
    for index in indices:
        for group in groups:
            for local in GROUP_LINEARS[group]:
                name = f"blocks.{index}.{local}"
                old = transformer.get_submodule(name)
                if isinstance(old, factory):
                    raise RuntimeError(f"{name} is already {factory.__name__}")
                if getattr(old, "input_rotation", None) is not None:
                    raise RuntimeError(f"{name} carries an input rotation; not combined here")
                weight = weight_source(f"{name}.weight").to(torch.bfloat16)
                bias = weight_source(f"{name}.bias").to(torch.bfloat16) if old.bias is not None else None
                if tuple(weight.shape) != (old.out_features, old.in_features):
                    raise ValueError(f"{name} checkpoint weight shape mismatch")
                device = next(t for t in (getattr(old, "weight_shuffle", None), old.weight, old.bias) if t is not None).device
                new = factory(old.in_features, old.out_features, bias=bias is not None, device="meta",
                              dtype=torch.bfloat16)
                new.load_and_quantize_weights(weight.to(device), None if bias is None else bias.to(device), device=device)
                _replace(transformer, name, new)
                replaced.append(name)
                del old, weight, bias
    return replaced
