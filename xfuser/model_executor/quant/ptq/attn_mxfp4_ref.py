"""Torch reference for the pinned AITER MHA v4 MXFP4 attention recipe.

The reference models numerical values, including the FP6-P reduction groups
and packed exponential approximation with exponent rollback. It does not
reproduce MFMA accumulation order. It is not bit exact.
Inputs and outputs use BSHD, matching ``aiter.ops.mha_v4.mha_v4``.
"""

from __future__ import annotations

import math

import torch

from .mxfp4_ref import decode_e8m0, dequantize_mxfp4, quantize_mxfp4


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
    """Round to finite FP6 E3M2 (historical probe, not pinned MHA v4 P)."""

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
    return _round_fp6(values, positive)


def _round_fp6(values: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
    magnitude = values.to(torch.float32).abs()
    boundaries = (positive[:-1] + positive[1:]) * 0.5
    code = torch.bucketize(magnitude, boundaries, right=False)
    bounded = code < boundaries.numel()
    lower = code.clamp(max=boundaries.numel() - 1)
    at_midpoint = bounded & (magnitude == boundaries[lower])
    code = code + (at_midpoint & ((lower & 1) == 1)).to(code.dtype)
    rounded = positive[code.clamp(max=positive.numel() - 1)]
    return torch.copysign(rounded, values.to(torch.float32))


def _fp6_p_groups(values: torch.Tensor) -> torch.Tensor:
    # gfx950 FP6-P groups alternate four-token chunks in each 64-key half:
    # {0:4, 8:12, ..., 56:60} and {4:8, 12:16, ..., 60:64}.
    sequence = values.shape[-1]
    padded = math.ceil(sequence / 64) * 64
    values = torch.nn.functional.pad(values, (0, padded - sequence))
    return values.reshape(*values.shape[:-1], -1, 8, 2, 4).transpose(-3, -2).reshape(
        *values.shape[:-1], -1, MX_BLOCK_SIZE
    )


def _ungroup_fp6_p(values: torch.Tensor, sequence: int) -> torch.Tensor:
    prefix = values.shape[:-2]
    return values.reshape(*prefix, -1, 2, 8, 4).transpose(-3, -2).reshape(
        *prefix, -1
    )[..., :sequence]


def quantize_mxfp6_e2m3(
    values: torch.Tensor, *, scale_from_bf16: bool = True
) -> torch.Tensor:
    """Convert P with one E8M0 scale per interleaved 32-key group.

    The pinned gfx950 object converts BF16 to E2M3, with scale exponent
    ``max(floor(log2(amax)) - 2, -126)``. Ragged groups are zero padded,
    without including the padding in the returned probability matrix.
    """

    sequence = values.shape[-1]
    grouped = _fp6_p_groups(values.to(torch.float32))
    bf16_grouped = grouped.to(torch.bfloat16).to(torch.float32)
    # Full tiles derive the scale from packed BF16. The standard-exp2 tail
    # derives it from FP32 before BF16 conversion (object 0xe7d4-0xe904).
    amax = (bf16_grouped if scale_from_bf16 else grouped).abs().amax(
        dim=-1, keepdim=True
    )
    # Extract the BF16 exponent as the object does; this also handles zero
    # groups without log2(0) and preserves exact powers of two.
    exponent = ((amax.contiguous().view(torch.int32) >> 23) & 0xFF) - 2
    scale = (exponent.clamp(min=1) << 23).contiguous().view(torch.float32)
    positive = torch.tensor(
        [m / 8 for m in range(8)]
        + [2 ** (e - 1) * (1 + m / 8) for e in range(1, 4) for m in range(8)],
        dtype=torch.float32,
        device=values.device,
    )
    converted = _round_fp6(bf16_grouped / scale, positive) * scale
    return _ungroup_fp6_p(converted, sequence)


def _attention_probability(logits_log2: torch.Tensor, *, quantized: bool):
    if quantized:
        unnormalized = _converted_kernel_probability(logits_log2)
        return unnormalized / unnormalized.sum(dim=-1, keepdim=True)
    shifted = logits_log2 - logits_log2.amax(dim=-1, keepdim=True)
    unnormalized = torch.exp2(shifted)
    return unnormalized / unnormalized.sum(dim=-1, keepdim=True)


def _converted_kernel_probability(logits_log2: torch.Tensor) -> torch.Tensor:
    """Model the normal dense FP6-P path, including a standard-exp2 tail.

    QK's K scales subtract nine exponent bits. A 128-key seed tile establishes
    a carried bias; UNORM16 codes are reinterpreted as BF16 P. Subsequent full
    tiles keep that bias (not a new max each tile). The partial final tile uses
    exp2 and rescales the previous numerator and converted-P denominator.
    """

    sequence = logits_log2.shape[-1]
    full_sequence = sequence // 128 * 128
    scores = logits_log2 / 512
    # Scalars s77/s26 in fwd_hd128_mxfp4.co at 0x19d8/0x19f0.
    magic = float.fromhex("0x1.ffc4p-4")
    exp_scale = float.fromhex("0x1.fffep+8")
    if full_sequence:
        seed_max = (scores[..., :128] + 1).amax(dim=-1, keepdim=True) - 1
        shifted = scores[..., :full_sequence] + (magic - seed_max)
        codes = torch.round(shifted.clamp(0, 1) * 65535).to(torch.int32)
        if bool((codes > 0x2FFC).any()):
            converted, seed_max = _packed_probability_rollback(
                scores[..., :full_sequence], seed_max
            )
        else:
            raw = (codes << 16).contiguous().view(torch.float32)
            converted = quantize_mxfp6_e2m3(raw)
    else:
        seed_max = scores.amax(dim=-1, keepdim=True)
        converted = scores[..., :0]

    if full_sequence != sequence:
        tail = scores[..., full_sequence:] + 1
        tail_max = torch.maximum(seed_max + 1, tail.amax(dim=-1, keepdim=True))
        offset = -tail_max * exp_scale
        # Match multiply-to-FP32 followed by FMA, rather than subtract then
        # multiply. Even a tiny cancellation error at the maximum can move
        # the FP32-derived tail scale across a power-of-two boundary.
        shifted_tail = (tail.double() * exp_scale + offset.double()).float()
        raw_tail = torch.exp2(shifted_tail) * 2**-63
        converted_tail = quantize_mxfp6_e2m3(raw_tail, scale_from_bf16=False)
        converted = converted * torch.exp2((seed_max + 1 - tail_max) * exp_scale)
        converted = torch.cat((converted, converted_tail), dim=-1)
    return converted


def _packed_probability_rollback(scores, bias):
    """Replay exponent-only rollback (object 0x9314-0x983c).

    Rollback shifts each affected query's BF16 codes by a whole exponent,
    rescales preceding converted P, and increases the carried score bias.
    Non-overflowing queries receive a zero shift even when their wave branches.
    Process contiguous spans between rollback events rather than every tile.
    """
    magic = float.fromhex("0x1.ffc4p-4")
    inverse_exp_scale = float.fromhex("0x1.0001p-9")
    converted = torch.empty_like(scores)
    position = 0
    while position < scores.shape[-1]:
        codes = torch.round(
            (scores[..., position:] + (magic - bias)).clamp(0, 1) * 65535
        ).to(torch.int32)
        maxima = codes.reshape(*codes.shape[:-1], -1, 128).amax(-1)
        overflow = maxima > 0x2FFC
        events = overflow.reshape(-1, overflow.shape[-1]).any(0).nonzero()
        if events.numel() == 0:
            raw = (codes << 16).contiguous().view(torch.float32)
            converted[..., position:] = quantize_mxfp6_e2m3(raw)
            break
        next_tile = int(events[0, 0])
        before = next_tile * 128
        if before:
            raw = (codes[..., :before] << 16).contiguous().view(torch.float32)
            converted[..., position:position + before] = quantize_mxfp6_e2m3(raw)
            position += before
        maximum = maxima[..., next_tile:next_tile + 1]
        # ceil((maximum - 0x1ffc) / 128), but only when code > 0x2ffc.
        shift = torch.where(maximum > 0x2FFC, (maximum - 0x1FFC + 127) >> 7, 0)
        factor = ((127 - shift).clamp(min=0) << 23).contiguous().view(torch.float32)
        converted[..., :position] *= factor
        current = codes[..., before:before + 128]
        current = (current - (shift << 7)).clamp(min=0)
        raw = (current << 16).contiguous().view(torch.float32)
        converted[..., position:position + 128] = quantize_mxfp6_e2m3(raw)
        bias = bias + shift.float() * inverse_exp_scale
        position += 128
    return converted, bias


def _mxfp4_rows(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    rows = values.reshape(-1, shape[-1]).contiguous()
    packed, scales = quantize_mxfp4(rows)
    return dequantize_mxfp4(packed, scales).reshape(shape)


def quantize_mxfp4_v(values: torch.Tensor) -> torch.Tensor:
    """Quantize V per channel over the FP6-P interleaved 32-key groups.

    Unlike Q/K, the pinned V quantizer nudges magnitudes down one FP32 ULP
    before conversion, so exact FP4 midpoints round toward zero.
    """

    if values.ndim != 4 or values.shape[-1] != HEAD_DIM:
        raise ValueError("V must be BSHD with head dimension 128")
    batch, sequence, heads, channels = values.shape
    channel_rows = values.permute(0, 2, 3, 1).to(torch.float32)
    grouped = _fp6_p_groups(channel_rows)
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    bits = amax.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 2 + ((bits & 0x7FFFFF) > 0x400000)
    scale = decode_e8m0(exponent.clamp(0, 255).to(torch.uint8))
    normalized = grouped / scale
    nudged = torch.nextafter(normalized, torch.zeros_like(normalized))
    positive = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=values.device,
    )
    quantized = _ungroup_fp6_p(_round_fp6(nudged, positive) * scale, sequence)
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
    probability = _attention_probability(logits_log2, quantized=quantize_p)
    quantized_v = quantize_mxfp4_v(value) if quantize_v else value.to(torch.float32)
    output = torch.einsum("bhqk,bkhd->bqhd", probability, quantized_v)
    return output.to(query.dtype)


def mxfp4_attention_decomposition(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_indices: torch.Tensor,
    softmax_scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Evaluate Q-only, K-only, V-only, P-only and full MXFP4 error arms.

    ``query_indices`` samples query rows while retaining the complete K/V
    context. This keeps the production softmax distribution without
    materializing a square production-length score matrix.
    """

    if query_indices.ndim != 1 or query_indices.numel() == 0:
        raise ValueError("query_indices must be a non-empty vector")
    if query_indices.min().item() < 0 or query_indices.max().item() >= query.shape[1]:
        raise ValueError("query_indices are outside the query sequence")

    scale = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
    sampled_query = query.index_select(1, query_indices.to(query.device))
    rotated_q = normalized_hadamard_128(sampled_query) * (
        scale * MHA_V4_LOG2E
    )
    rotated_q = rotated_q.to(query.dtype).to(torch.float32)
    quantized_q = _mxfp4_rows(rotated_q)

    rotated_k = normalized_hadamard_128(key).to(key.dtype).to(torch.float32)
    quantized_key_input = key
    mean = aiter_k_mean(key)
    if mean is not None:
        quantized_key_input = key.to(torch.float32) - mean.unsqueeze(1)
    quantized_k = _mxfp4_rows(
        normalized_hadamard_128(quantized_key_input).to(key.dtype).to(torch.float32)
    )
    quantized_v = quantize_mxfp4_v(value)

    def probability(q_values, k_values, *, quantized):
        logits_log2 = torch.einsum("bqhd,bkhd->bhqk", q_values, k_values)
        return _attention_probability(logits_log2, quantized=quantized)

    probability_ref = probability(rotated_q, rotated_k, quantized=False)
    probability_q = probability(quantized_q, rotated_k, quantized=False)
    probability_k = probability(rotated_q, quantized_k, quantized=False)
    probability_p = probability(rotated_q, rotated_k, quantized=True)
    probability_all = probability(quantized_q, quantized_k, quantized=True)

    def attend(probabilities, values):
        return torch.einsum(
            "bhqk,bkhd->bqhd", probabilities, values.to(torch.float32)
        ).to(query.dtype)

    return {
        "reference": attend(probability_ref, value),
        "q_only": attend(probability_q, value),
        "k_only": attend(probability_k, value),
        "v_only": attend(probability_ref, quantized_v),
        "p_only": attend(probability_p, value),
        "all": attend(probability_all, quantized_v),
    }
