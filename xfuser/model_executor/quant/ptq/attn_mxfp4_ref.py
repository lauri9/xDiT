"""Torch reference for the pinned AITER MHA v4 MXFP4 attention recipe.

The reference models numerical values, not AITER's packed tile layouts. Inputs
and outputs use BSHD, matching ``aiter.ops.mha_v4.mha_v4``.
"""

from __future__ import annotations

import math

import torch

from .mxfp4_ref import dequantize_mxfp4, quantize_mxfp4


HEAD_DIM = 128
MX_BLOCK_SIZE = 32
K_SMOOTH_MIN_COMMON = 0.85
K_SMOOTH_SAMPLE_ROWS = 2048
MHA_V4_LOG2E = math.log2(math.e)


def normalized_hadamard_128(values: torch.Tensor) -> torch.Tensor:
    """Apply AITER's normalized Walsh-Hadamard transform to hd128 rows."""

    if values.shape[-1] != HEAD_DIM:
        raise ValueError("MHA v4 MXFP4 requires head dimension 128")
    result = values.to(torch.float32)
    width = 1
    while width < HEAD_DIM:
        pairs = result.reshape(*result.shape[:-1], -1, width * 2)
        left = pairs[..., :width].clone()
        right = pairs[..., width:].clone()
        result = torch.cat((left + right, left - right), dim=-1).reshape(
            *result.shape
        )
        width *= 2
    return result * (HEAD_DIM**-0.5)


def quantize_fp6_e3m2(values: torch.Tensor) -> torch.Tensor:
    """Round to finite FP6 E3M2, the MHA v4 probability operand format."""

    positive = torch.tensor(
        [
            0.0,
            0.0625,
            0.125,
            0.1875,
            0.25,
            0.3125,
            0.375,
            0.4375,
            0.5,
            0.625,
            0.75,
            0.875,
            1.0,
            1.25,
            1.5,
            1.75,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            10.0,
            12.0,
            14.0,
            16.0,
            20.0,
            24.0,
            28.0,
        ],
        dtype=torch.float32,
        device=values.device,
    )
    magnitude = values.to(torch.float32).abs()
    boundaries = (positive[:-1] + positive[1:]) * 0.5
    code = torch.bucketize(magnitude, boundaries, right=False)
    bounded = code < boundaries.numel()
    lower = code.clamp(max=boundaries.numel() - 1)
    at_midpoint = bounded & (magnitude == boundaries[lower])
    code = code + (at_midpoint & ((lower & 1) == 1)).to(code.dtype)
    rounded = positive[code.clamp(max=positive.numel() - 1)]
    return torch.copysign(rounded, values.to(torch.float32))


def _mxfp4_rows(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    rows = values.reshape(-1, shape[-1]).contiguous()
    packed, scales = quantize_mxfp4(rows)
    return dequantize_mxfp4(packed, scales).reshape(shape)


def quantize_mxfp4_v(values: torch.Tensor) -> torch.Tensor:
    """Quantize V per channel over blocks of 32 sequence positions."""

    if values.ndim != 4 or values.shape[-1] != HEAD_DIM:
        raise ValueError("V must be BSHD with head dimension 128")
    batch, sequence, heads, channels = values.shape
    padded_sequence = math.ceil(sequence / MX_BLOCK_SIZE) * MX_BLOCK_SIZE
    channel_rows = values.permute(0, 2, 3, 1).to(torch.float32)
    if padded_sequence != sequence:
        channel_rows = torch.nn.functional.pad(
            channel_rows, (0, padded_sequence - sequence)
        )
    quantized = _mxfp4_rows(channel_rows)[..., :sequence]
    return quantized.permute(0, 3, 1, 2).reshape(
        batch, sequence, heads, channels
    )


def aiter_k_mean(key: torch.Tensor) -> torch.Tensor | None:
    """Reproduce the pinned wrapper's gated, sampled K-centering decision."""

    stride = max(1, key.shape[1] // K_SMOOTH_SAMPLE_ROWS)
    sample = key[:, ::stride].to(torch.float32)
    mean = sample.mean(dim=1)
    row_sq = sample.square().sum(dim=-1).mean(dim=1)
    gate = mean.square().sum(dim=-1) > K_SMOOTH_MIN_COMMON**2 * row_sq
    if not bool(gate.any()):
        return None
    return mean * gate.unsqueeze(-1)


def mxfp4_attention_ref(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    quantize_q: bool = True,
    quantize_k: bool = True,
    quantize_v: bool = True,
    quantize_p: bool = True,
    apply_aiter_k_centering: bool = True,
) -> torch.Tensor:
    """Emulate dense, non-causal MHA v4 MXFP4 from BF16 BSHD operands."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Q, K, and V must be BSHD tensors")
    if query.shape[-1] != HEAD_DIM:
        raise ValueError("MHA v4 MXFP4 requires head dimension 128")
    if key.shape != value.shape:
        raise ValueError("K and V must have matching shapes")
    if query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]:
        raise ValueError("the reference currently requires matching batch and heads")

    scale = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
    centered_key = key
    if apply_aiter_k_centering and quantize_k:
        mean = aiter_k_mean(key)
        if mean is not None:
            centered_key = key.to(torch.float32) - mean.unsqueeze(1)

    rotated_q = normalized_hadamard_128(query) * (scale * MHA_V4_LOG2E)
    rotated_k = normalized_hadamard_128(centered_key)
    # The fused AITER quantizer rounds the rotated values through the input
    # floating-16 type before deriving E8M0 scales.
    rotated_q = rotated_q.to(query.dtype).to(torch.float32)
    rotated_k = rotated_k.to(key.dtype).to(torch.float32)
    if quantize_q:
        rotated_q = _mxfp4_rows(rotated_q)
    if quantize_k:
        rotated_k = _mxfp4_rows(rotated_k)

    logits_log2 = torch.einsum("bqhd,bkhd->bhqk", rotated_q, rotated_k)
    shifted = logits_log2 - logits_log2.amax(dim=-1, keepdim=True)
    unnormalized_probability = torch.exp2(shifted)
    denominator = unnormalized_probability.sum(dim=-1, keepdim=True)
    if quantize_p:
        unnormalized_probability = quantize_fp6_e3m2(unnormalized_probability)
    probability = unnormalized_probability / denominator
    quantized_v = quantize_mxfp4_v(value) if quantize_v else value.to(torch.float32)
    output = torch.einsum("bhqk,bkhd->bqhd", probability, quantized_v)
    return output.to(query.dtype)
