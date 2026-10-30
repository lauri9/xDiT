"""Install online GEMM-input rotations into a loaded Wan transformer.

For each block and each rotated group, the consuming linears get ``W ← W R``
in their stored dtype and ``input_rotation = R``; their forward then applies
``R`` to the activation before its MXFP4 quantizer. This must run before the
first forward of an RTN ``xFuserMXFP4Linear``, which quantizes its BF16
weight lazily, so the packed weight is the rotated one.
"""
from __future__ import annotations

import torch

from .rotations import Rotation, fold_input

TARGETS = {
    "self_qkv": ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    "self_out": ("attn1.to_out.0",),
    "cross_q": ("attn2.to_q",),
    "cross_out": ("attn2.to_out.0",),
    "ffn_in": ("ffn.net.0.proj",),
    "ffn_out": ("ffn.net.2",),
}


def _targets(block, group):
    if group == "self_qkv" and getattr(block.attn1, "fused_projections", False):
        return ("attn1.to_qkv",)
    return TARGETS[group]


@torch.no_grad()
def install_online_rotations(transformer, online: dict[str, Rotation], *, blocks=None) -> list[str]:
    """Fold and attach rotations; returns the module names that were changed."""
    unknown = set(online) - set(TARGETS)
    if unknown:
        raise ValueError(f"no online rotation site for {sorted(unknown)}")
    indices = range(len(transformer.blocks)) if blocks is None else blocks
    changed = []
    for index in indices:
        block = transformer.blocks[index]
        for group, rotation in online.items():
            for name in _targets(block, group):
                module = block.get_submodule(name)
                if not hasattr(module, "input_rotation"):
                    raise TypeError(f"blocks.{index}.{name} is {type(module).__name__}, not a rotatable MXFP4 linear")
                if getattr(module, "weight", None) is None or hasattr(module, "weight_shuffle"):
                    raise RuntimeError(f"blocks.{index}.{name} is already quantized; install before the first forward")
                if module.input_rotation is not None:
                    raise RuntimeError(f"blocks.{index}.{name} already has an input rotation")
                if module.weight.shape[1] != rotation.dim:
                    raise ValueError(f"blocks.{index}.{name} input width does not match the rotation")
                module.weight.data.copy_(fold_input(module.weight.data, rotation).to(module.weight.dtype))
                module.input_rotation = rotation
                changed.append(f"blocks.{index}.{name}")
    return changed
