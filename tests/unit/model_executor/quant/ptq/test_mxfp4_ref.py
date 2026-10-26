import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.mxfp4_ref import (
    decode_e8m0,
    dequantize_mxfp4,
    encode_e8m0,
    pack_fp4,
    quantize_mxfp4,
    unpack_fp4,
)


def test_pack_unpack_round_trip():
    unpacked = torch.arange(16, dtype=torch.uint8).repeat(2, 2)

    restored = unpack_fp4(pack_fp4(unpacked))

    torch.testing.assert_close(restored, unpacked, rtol=0, atol=0)


def test_e8m0_encoding_is_a_non_saturating_power_of_two_scale():
    amax = torch.tensor([0.0, 0.5, 6.0, 6.01, 12.0], dtype=torch.float32)

    scale = decode_e8m0(encode_e8m0(amax))

    assert torch.all(scale > 0)
    assert torch.all(scale[1:] * 6 >= amax[1:])
    torch.testing.assert_close(torch.log2(scale[1:]).round(), torch.log2(scale[1:]))


def test_quantize_dequantize_uses_per_row_per_32_scales():
    values = torch.zeros((2, 64), dtype=torch.float32)
    values[0, :32] = 0.5
    values[0, 32:] = 12.0
    values[1] = torch.linspace(-6.0, 6.0, 64)

    packed, scales = quantize_mxfp4(values)
    restored = dequantize_mxfp4(packed, scales)

    assert packed.shape == (2, 32)
    assert scales.shape == (2, 2)
    torch.testing.assert_close(restored[0], values[0], rtol=0, atol=0)
    assert torch.isfinite(restored).all()


def test_e2m1_midpoints_round_to_even_codes():
    values = torch.zeros((1, 32), dtype=torch.float32)
    values[0, :7] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    values[0, 7:14] = -values[0, :7]

    packed, scales = quantize_mxfp4(values)
    restored = dequantize_mxfp4(packed, scales)
    scale = decode_e8m0(scales)[0, 0]

    expected = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]) * scale
    torch.testing.assert_close(restored[0, :7], expected, rtol=0, atol=0)
    torch.testing.assert_close(restored[0, 7:14], -expected, rtol=0, atol=0)


def test_invalid_shapes_are_rejected():
    with pytest.raises(ValueError, match="two-dimensional"):
        quantize_mxfp4(torch.zeros(32))
    with pytest.raises(ValueError, match="divisible by 32"):
        quantize_mxfp4(torch.zeros((1, 31)))
