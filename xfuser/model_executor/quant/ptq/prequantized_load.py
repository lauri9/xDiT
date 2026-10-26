"""Meta-layout and manifest seam for pre-quantized AITER MXFP4 weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn

from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear
from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
    DerivedTensor,
)


@dataclass(frozen=True)
class PackedLinearSpec:
    """Artifact mapping for one component-relative linear."""

    module_name: str
    packed_key: str
    scale_key: str
    packed_shape: tuple[int, int]
    shuffled_scale_shape: tuple[int, int]
    artifact_path: str | None = None


def _replace_submodule(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def install_prequantized_mxfp4_layout(
    model: nn.Module,
    specs: Iterable[PackedLinearSpec],
) -> None:
    """Replace artifact-owned meta linears with their final packed layout."""

    for spec in specs:
        linear = model.get_submodule(spec.module_name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(
                f"pre-quantized MXFP4 target {spec.module_name!r} is "
                f"{type(linear).__name__}, expected an unfused nn.Linear"
            )
        replacement = xFuserMXFP4Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        replacement.install_packed_placeholders(
            weight_shape=spec.packed_shape,
            scale_shape=spec.shuffled_scale_shape,
        )
        replacement.train(linear.training)
        _replace_submodule(model, spec.module_name, replacement)


def _shuffle_weight(raw: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_weight

    packed_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    return shuffle_weight(raw.contiguous().view(packed_dtype), layout=(16, 16))


def _shuffle_scale(raw: torch.Tensor) -> torch.Tensor:
    from aiter.ops.shuffle import shuffle_scale

    scale_dtype = getattr(torch, "float8_e8m0fnu", torch.uint8)
    return shuffle_scale(raw.contiguous().view(scale_dtype))


def compose_prequantized_manifest(
    base: CheckpointManifest,
    *,
    artifact_path: str | None,
    specs: Iterable[PackedLinearSpec],
    weight_builder: Callable[[torch.Tensor], torch.Tensor] = _shuffle_weight,
    scale_builder: Callable[[torch.Tensor], torch.Tensor] = _shuffle_scale,
) -> CheckpointManifest:
    """Overlay packed artifact tensors while retaining BF16 biases and tails."""

    weight_map = dict(base.weight_map)
    checkpoint_keys = dict(base.checkpoint_keys)
    derived = dict(base.derived)

    for spec in specs:
        source_path = spec.artifact_path or artifact_path
        if source_path is None:
            raise ValueError(f"artifact path is missing for {spec.module_name!r}")
        weight_key = f"{spec.module_name}.weight"
        packed_live_key = f"{spec.module_name}.weight_shuffle"
        scale_live_key = f"{spec.module_name}.weight_scale"
        if weight_key not in weight_map:
            raise ValueError(f"BF16 checkpoint has no target weight {weight_key!r}")
        weight_map.pop(weight_key)
        checkpoint_keys.pop(weight_key, None)
        derived.pop(weight_key, None)

        weight_map[packed_live_key] = source_path
        weight_map[scale_live_key] = source_path
        derived[packed_live_key] = DerivedTensor(
            sources=(spec.packed_key,),
            build=weight_builder,
            description="shuffle unshuffled MXFP4 packed weight",
        )
        derived[scale_live_key] = DerivedTensor(
            sources=(spec.scale_key,),
            build=scale_builder,
            description="pad and shuffle unshuffled E8M0 scales",
        )

    return CheckpointManifest(
        weight_map=weight_map,
        checkpoint_keys=checkpoint_keys,
        strict=True,
        label=f"pre-quantized overlay {artifact_path or 'sharded artifact'}",
        derived=derived,
        sparse_paths=base.shard_paths | base.sparse_paths,
    )
