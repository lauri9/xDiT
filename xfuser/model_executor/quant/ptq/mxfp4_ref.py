"""Dependency-free torch reference for AITER's per-1x32 MXFP4 format."""

from __future__ import annotations

import torch


MXFP4_BLOCK_SIZE = 32
_FP4_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_FP4_BOUNDARIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_MIN_AMAX = 1.0e-10


def encode_e8m0(amax: torch.Tensor) -> torch.Tensor:
    """Encode ``ceil_pow2(max(amax, 1e-10) / 6)`` as an E8M0 byte."""

    values = torch.clamp(amax.to(torch.float32), min=_MIN_AMAX) / 6.0
    bits = values.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa_nonzero = (bits & 0x7FFFFF) != 0
    exponent = exponent + (mantissa_nonzero & (exponent < 0xFF)).to(torch.int32)
    return exponent.to(torch.uint8)


def decode_e8m0(encoded: torch.Tensor) -> torch.Tensor:
    """Decode E8M0 bytes to their float32 power-of-two values."""

    encoded_u8 = encoded.to(torch.uint8)
    bits = encoded_u8.to(torch.int32) << 23
    bits = torch.where(encoded_u8 == 0, torch.full_like(bits, 0x00400000), bits)
    bits = torch.where(encoded_u8 == 0xFF, torch.full_like(bits, 0x7F800001), bits)
    return bits.contiguous().view(torch.float32)


def _quantize_e2m1_unpacked(values: torch.Tensor) -> torch.Tensor:
    """Return one E2M1 code per value, including the sign nibble."""

    values_f32 = values.to(torch.float32)
    magnitude = values_f32.abs()
    boundaries = torch.tensor(
        _FP4_BOUNDARIES,
        dtype=torch.float32,
        device=values.device,
    )
    code = torch.bucketize(magnitude, boundaries, right=False)

    # E2M1 uses round-to-nearest-even. At an exact midpoint, choose the
    # endpoint whose packed magnitude code is even.
    bounded = code < len(_FP4_BOUNDARIES)
    boundary_index = code.clamp(max=len(_FP4_BOUNDARIES) - 1)
    at_midpoint = bounded & (
        magnitude == boundaries[boundary_index]
    )
    code = code + (at_midpoint & ((boundary_index & 1) == 1)).to(code.dtype)
    code = code.clamp(max=7).to(torch.uint8)
    sign = torch.signbit(values_f32).to(torch.uint8) << 3
    return code | sign


def pack_fp4(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack adjacent low/high FP4 nibbles into uint8 bytes."""

    if unpacked.shape[-1] % 2:
        raise ValueError("the FP4 dimension must be even")
    values = unpacked.to(torch.uint8)
    return (values[..., ::2] & 0xF) | ((values[..., 1::2] & 0xF) << 4)


def unpack_fp4(packed: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 bytes into low-nibble-first FP4 codes."""

    raw = packed.to(torch.uint8)
    shape = (*raw.shape[:-1], raw.shape[-1] * 2)
    unpacked = torch.empty(shape, dtype=torch.uint8, device=raw.device)
    unpacked[..., ::2] = raw & 0xF
    unpacked[..., 1::2] = raw >> 4
    return unpacked


def _decode_fp4(unpacked: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_FP4_POSITIVE, dtype=torch.float32, device=unpacked.device)
    magnitude = table[(unpacked & 0x7).long()]
    return torch.where((unpacked & 0x8) != 0, -magnitude, magnitude)


def quantize_mxfp4(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a two-dimensional tensor to unshuffled packed MXFP4."""

    if values.ndim != 2:
        raise ValueError("MXFP4 reference expects a two-dimensional tensor")
    rows, columns = values.shape
    if columns % MXFP4_BLOCK_SIZE:
        raise ValueError("the MXFP4 K dimension must be divisible by 32")

    grouped = values.to(torch.float32).reshape(
        rows, columns // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE
    )
    scales = encode_e8m0(grouped.abs().amax(dim=-1))
    scale_f32 = decode_e8m0(scales).unsqueeze(-1)
    unpacked = _quantize_e2m1_unpacked(grouped / scale_f32)
    return pack_fp4(unpacked.reshape(rows, columns)), scales


def dequantize_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Dequantize an unshuffled packed MXFP4 tensor to float32."""

    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed values and scales must both be two-dimensional")
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    if columns % MXFP4_BLOCK_SIZE:
        raise ValueError("the MXFP4 K dimension must be divisible by 32")
    if tuple(scales.shape) != (rows, columns // MXFP4_BLOCK_SIZE):
        raise ValueError("scale shape does not match the packed MXFP4 tensor")

    values = _decode_fp4(unpack_fp4(packed)).reshape(
        rows, columns // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE
    )
    return (values * decode_e8m0(scales).unsqueeze(-1)).reshape(rows, columns)
