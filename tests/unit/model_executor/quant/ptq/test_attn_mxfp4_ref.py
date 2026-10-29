import math

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.attn_mxfp4_ref import (
    aiter_k_mean,
    mxfp4_attention_decomposition,
    mxfp4_attention_ref,
    normalized_hadamard_128,
    quantize_fp6_e3m2,
    quantize_mxfp6_e2m3,
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
    values = torch.zeros((1, 65, 1, 128), dtype=torch.bfloat16)
    even = torch.arange(64).remainder(8) < 4
    values[:, :64, :, 0] = torch.where(even, 0.5, 12.0).reshape(1, 64, 1)
    values[:, 64:, :, 0] = 3.0
    values[:, :, :, 1] = 3.0

    actual = quantize_mxfp4_v(values)

    torch.testing.assert_close(actual[:, :64, :, 0], values[:, :64, :, 0].float())
    torch.testing.assert_close(actual[:, 64:, :, 0], torch.full((1, 1, 1), 3.0))
    torch.testing.assert_close(actual[:, :, :, 1], torch.full((1, 65, 1), 3.0))


def test_p_conversion_scales_each_key_block_and_rounds_to_even():
    values = torch.zeros((2, 65))
    values[0, 0] = 1.0
    values[0, [1, 2, 3, 8]] = torch.tensor([0.015625, 0.046875, 0.265625, 0.296875])
    odd = torch.arange(64).remainder(8) >= 4
    values[0, :64][odd] = 2.0**-10
    values[0, 64] = 2.0**-20
    values[1] = values[0] * 8

    actual = quantize_mxfp6_e2m3(values)

    torch.testing.assert_close(
        actual[0, [0, 1, 2, 3, 8]], torch.tensor([1.0, 0.0, 0.0625, 0.25, 0.3125]),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(actual[0, :64][odd], values[0, :64][odd], rtol=0, atol=0)
    torch.testing.assert_close(actual[0, 64:], values[0, 64:], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], actual[0] * 8, rtol=0, atol=0)
    assert torch.isfinite(quantize_mxfp6_e2m3(torch.zeros((1, 32)))).all()


def test_v_midpoints_round_toward_zero_in_both_signs():
    values = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    rows = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])
    values[0, rows, 0, 0] = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], dtype=values.dtype,
    )
    values[..., 1] = -values[..., 0]

    actual = quantize_mxfp4_v(values)

    expected = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    torch.testing.assert_close(actual[0, rows, 0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0, rows, 0, 1], -expected, rtol=0, atol=0)


def test_unmodeled_probability_overflow_rollback_is_rejected():
    query = torch.ones((1, 1, 1, 128), dtype=torch.bfloat16)
    key = torch.zeros((1, 256, 1, 128), dtype=torch.bfloat16)
    key[:, 200] = 8

    with pytest.raises(NotImplementedError, match="overflow rollback"):
        mxfp4_attention_ref(query, key, torch.zeros_like(key))


@pytest.mark.parametrize("sequence", [33, 4096])
def test_attention_and_decomposition_preserve_constant_values(sequence):
    generator = torch.Generator().manual_seed(20261029 + sequence)
    query = torch.randn((1, 3, 2, 128), generator=generator).mul(2**0.5).bfloat16()
    key = torch.randn((1, sequence, 2, 128), generator=generator).mul(2**0.5).bfloat16()
    channels = torch.linspace(-1, 1, 128).bfloat16()
    value = channels.expand_as(key).clone()
    indices = torch.tensor([0, 2])

    actual = mxfp4_attention_ref(query, key, value, quantize_v=False)
    outputs = mxfp4_attention_decomposition(query, key, value, query_indices=indices)

    torch.testing.assert_close(actual, channels.expand_as(query), rtol=0, atol=0)
    torch.testing.assert_close(
        outputs["p_only"], channels.expand_as(outputs["p_only"]), rtol=0, atol=0,
    )
    expected_all = mxfp4_attention_ref(query.index_select(1, indices), key, value)
    torch.testing.assert_close(outputs["all"], expected_all, rtol=0, atol=0)


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


def test_decomposition_samples_queries_against_the_full_kv_context():
    torch.manual_seed(4)
    query = torch.randn((1, 9, 2, 128), dtype=torch.bfloat16)
    key = torch.randn((1, 11, 2, 128), dtype=torch.bfloat16)
    value = torch.randn_like(key)
    indices = torch.tensor([0, 4, 8])

    outputs = mxfp4_attention_decomposition(
        query,
        key,
        value,
        query_indices=indices,
    )

    assert set(outputs) == {
        "reference",
        "q_only",
        "k_only",
        "v_only",
        "p_only",
        "all",
    }
    assert all(output.shape == (1, 3, 2, 128) for output in outputs.values())
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.index_select(1, indices).transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
    ).transpose(1, 2)
    torch.testing.assert_close(
        outputs["reference"], expected, rtol=2.0e-2, atol=2.0e-2
    )
