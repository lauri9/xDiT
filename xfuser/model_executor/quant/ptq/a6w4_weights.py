"""AITER A6W4 weight domain: H32-rotated MXFP4 codes.

AITER's gfx950 A6W4 GEMM applies a block-diagonal 32x32 Hadamard (BF16-rounded
H, the same matrix for activations and weights) along K inside its packers:
it computes ``Q6(x H) · Q4(bf16(W H))ᵀ``. Production A6W4 "RTN" is therefore
RTN of the *rotated* weight. A GPTQ for A6W4 must work in that domain, on
``bf16(W H)`` with the Hessian ``Hᵀ (XᵀX) H``, and its codes must be packed
directly into the kernel tile layout; re-packing dequantized weights through
``quant_mxfp4_gemm`` would rotate them a second time.

All AITER symbols are imported lazily (GPU-only package).
"""
from __future__ import annotations

import torch

K_GROUP = 32
WEIGHT_DOMAIN = "a6w4_h32"


def _a6w6():
    import aiter.ops.gemm_op_a6w6 as module

    return module


def _a6w4():
    import aiter.ops.gemm_op_a6w4 as module

    return module


def kernel_h32(device) -> torch.Tensor:
    """The kernel's BF16-rounded 32x32 Hadamard as FP32 (right-multiplied)."""
    return _a6w6()._had32_t(torch.device(device)).to(torch.bfloat16).float()


def rotated_weight(weight: torch.Tensor) -> torch.Tensor:
    """``bf16(W H)`` as FP32: exactly what AITER's MXFP4 packer quantizes."""
    return _a6w6()._rotate_k32_torch(weight).to(torch.bfloat16).float()


def conjugate_hessian(hessian: torch.Tensor, h32: torch.Tensor) -> torch.Tensor:
    """``Bᵀ H B`` for the block-diagonal ``B = diag(h32, ..., h32)``."""
    k = hessian.shape[0]
    if hessian.shape != (k, k) or k % K_GROUP:
        raise ValueError("Hessian must be square with K divisible by 32")
    blocks = hessian.float().reshape(k // K_GROUP, K_GROUP, k // K_GROUP, K_GROUP)
    h32 = h32.to(device=hessian.device, dtype=torch.float32)
    rotated = torch.einsum("ac,iajb,bd->icjd", h32, blocks, h32).reshape(k, k)
    return 0.5 * (rotated + rotated.T)


def effective_weight(packed: torch.Tensor, scale: torch.Tensor, h32: torch.Tensor) -> torch.Tensor:
    """Unrotated weight the kernel effectively applies, ``deq(codes) Bᵀ`` (FP32)."""
    from .mxfp4_ref import dequantize_mxfp4

    values = dequantize_mxfp4(packed, scale)
    rows, k = values.shape
    return (values.reshape(rows, k // K_GROUP, K_GROUP) @ h32.to(values.device).T).reshape(rows, k)


def pack_codes(packed: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural ``[R, K/2]`` E2M1 bytes and ``[R, K/32]`` E8M0 scales (rotated
    domain) -> the A6W4 kernel's compact packed and scale blobs."""
    rows, half_k = packed.shape
    if rows % 256 or (2 * half_k) % 128 or tuple(scale.shape) != (rows, 2 * half_k // K_GROUP):
        raise ValueError("A6W4 code packing needs rows % 256 == 0, K % 128 == 0 and matching scales")
    blob = _a6w4()._pack_mxfp4_gemm_torch(packed.contiguous())
    scale_blob = _a6w6().pack_scale_torch(scale.contiguous().view(torch.uint8), rows)
    return blob, scale_blob


def load_packed(layer, blob: torch.Tensor, scale_blob: torch.Tensor, bias: torch.Tensor | None) -> None:
    """Install packed A6W4 blobs into an ``xFuserA6W4Linear`` (validated sizes)."""
    from xfuser.model_executor.layers.mixed_mxfp_linear import _validate_packed_pair

    _validate_packed_pair(blob, scale_blob, rows=layer.out_features, in_features=layer.in_features,
                          format_=layer.weight_format, name="A6W4 GPTQ weight")
    if bias is not None:
        layer.bias = torch.nn.Parameter(bias.detach().to(device=blob.device, dtype=torch.bfloat16))
    if hasattr(layer, "weight"):
        delattr(layer, "weight")
    layer.register_parameter("weight", None)
    layer._remove_packed_state()
    layer.register_parameter("weight_packed", torch.nn.Parameter(blob, requires_grad=False))
    layer.register_buffer("weight_scale", scale_blob, persistent=True)
    layer._compute_dtype = torch.bfloat16


def artifact_code_source(artifact_root, component: str, device="cuda"):
    """``module_name -> (packed, scale)`` from an A6W4-domain GPTQ artifact."""
    from safetensors import safe_open

    from .artifact import load_artifact
    from .calibrate import MODEL_REVISION

    artifact = load_artifact(artifact_root, expected_source_revision=MODEL_REVISION)
    domain = artifact.config.get("gptq_config", {}).get("weight_domain", "identity")
    if domain != WEIGHT_DOMAIN:
        raise ValueError(f"artifact {artifact.artifact_id} is in weight domain {domain!r}, not {WEIGHT_DOMAIN}")
    layers = artifact.config["layers"]

    def read(name: str):
        entry = layers.get(f"{component}.{name}")
        if entry is None:
            raise KeyError(f"artifact {artifact.artifact_id} does not cover {component}.{name}")
        with safe_open(artifact.root / entry["shard"], framework="pt", device="cpu") as handle:
            return handle.get_tensor(entry["packed_key"]).to(device), handle.get_tensor(entry["scale_key"]).to(device)

    read.artifact_id = artifact.artifact_id
    return read


@torch.no_grad()
def install_a6w4_codes(transformer, groups, *, code_source, bias_source, blocks=None) -> list[str]:
    """Replace ``groups`` linears with ``xFuserA6W4Linear`` holding given codes."""
    from xfuser.model_executor.layers.mixed_mxfp_linear import xFuserA6W4Linear

    from .selective_precision import GROUP_LINEARS, _replace

    unknown = set(groups) - set(GROUP_LINEARS)
    if unknown or not groups:
        raise ValueError(f"unknown or empty GEMM groups: {sorted(unknown) or groups}")
    replaced = []
    for index in range(len(transformer.blocks)) if blocks is None else blocks:
        for group in groups:
            for local in GROUP_LINEARS[group]:
                name = f"blocks.{index}.{local}"
                old = transformer.get_submodule(name)
                packed, scale = code_source(name)
                if tuple(packed.shape) != (old.out_features, old.in_features // 2):
                    raise ValueError(f"{name} code shape mismatch")
                bias = bias_source(f"{name}.bias") if old.bias is not None else None
                new = xFuserA6W4Linear(old.in_features, old.out_features, bias=bias is not None, device="meta",
                                       dtype=torch.bfloat16)
                load_packed(new, *pack_codes(packed, scale), bias)
                _replace(transformer, name, new)
                replaced.append(name)
    return replaced
