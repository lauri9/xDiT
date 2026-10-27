import math

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.attn_mxfp4_ref import (
    aiter_k_mean,
    mxfp4_attention_ref,
    normalized_hadamard_128,
    quantize_fp6_e3m2,
    quantize_mxfp4_v,
)


def test_normalized_hadamard_preserves_inner_products():
    torch.manual_seed(1)
    left = torch.randn(3, 128)
    right = torch.randn(3, 128)

    expected = (left * right).sum(dim=-1)
    actual = (
        normalized_hadamard_128(left) * normalized_hadamard_128(right)
    ).sum(dim=-1)

    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-5)


def test_fp6_e3m2_uses_round_to_nearest_even():
    values = torch.tensor([0.03125, 0.09375, 0.15625, 0.21875, 1.125, 30.0])

    actual = quantize_fp6_e3m2(values)

    torch.testing.assert_close(
        actual,
        torch.tensor([0.0, 0.125, 0.125, 0.25, 1.0, 28.0]),
        rtol=0,
        atol=0,
    )


def test_v_quantization_blocks_the_sequence_axis_per_channel():
    values = torch.zeros((1, 33, 1, 128), dtype=torch.bfloat16)
    values[:, :32, :, 0] = 0.5
    values[:, 32:, :, 0] = 12.0
    values[:, :, :, 1] = 3.0

    actual = quantize_mxfp4_v(values)

    torch.testing.assert_close(actual[:, :32, :, 0], torch.full((1, 32, 1), 0.5))
    torch.testing.assert_close(actual[:, 32:, :, 0], torch.full((1, 1, 1), 12.0))
    torch.testing.assert_close(actual[:, :, :, 1], torch.full((1, 33, 1), 3.0))


def test_aiter_k_centering_gate_rejects_normal_inputs_and_accepts_common_mode():
    torch.manual_seed(2)
    ordinary = torch.randn((1, 64, 2, 128), dtype=torch.bfloat16)
    common = torch.ones((1, 64, 2, 128), dtype=torch.bfloat16)

    assert aiter_k_mean(ordinary) is None
    mean = aiter_k_mean(common)

    assert mean is not None
    torch.testing.assert_close(mean, torch.ones_like(mean))


def test_attention_without_quantization_matches_sdpa():
    torch.manual_seed(3)
    query = torch.randn((1, 7, 2, 128), dtype=torch.bfloat16)
    key = torch.randn((1, 9, 2, 128), dtype=torch.bfloat16)
    value = torch.randn_like(key)

    actual = mxfp4_attention_ref(
        query,
        key,
        value,
        quantize_q=False,
        quantize_k=False,
        quantize_v=False,
        quantize_p=False,
        apply_aiter_k_centering=False,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        scale=1.0 / math.sqrt(128),
    ).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=2.0e-2, atol=2.0e-2)


def test_invalid_attention_shapes_are_rejected():
    query = torch.zeros((1, 2, 1, 64), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="head dimension 128"):
        mxfp4_attention_ref(query, query, query)
