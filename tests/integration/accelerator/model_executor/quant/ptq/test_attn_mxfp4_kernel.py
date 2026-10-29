import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.quant.ptq.attn_mxfp4_ref import mxfp4_attention_ref


pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


def _mha_v4_mxfp4(query, key, value):
    module = pytest.importorskip("aiter.ops.mha_v4")
    return module.mha_v4(
        query,
        key,
        value,
        module.AttentionFormat.MXFP4,
        module.AttentionFormat.MXFP4,
        module.AttentionFormat.MXFP4,
    )


@pytest.mark.parametrize("sequence", [128, 208, 4096, 16384, 75600])
def test_torch_emulator_matches_mha_v4_mxfp4(sequence):
    generator = torch.Generator(device="cuda").manual_seed(20261027 + sequence)
    shape = (1, sequence, 2, 128)
    query = torch.randn(
        shape, dtype=torch.float32, device="cuda", generator=generator
    ).mul_(2**0.5).bfloat16()
    key = torch.randn(
        shape, dtype=torch.float32, device="cuda", generator=generator
    ).mul_(2**0.5).bfloat16()
    value = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )

    with torch.no_grad():
        # Full dense kernel, sampled queries with complete K/V in the reference.
        rows = torch.linspace(0, sequence - 1, min(sequence, 64), device="cuda").long()
        expected = _mha_v4_mxfp4(query, key, value).index_select(1, rows)
        sampled_query = query.index_select(1, rows)
        actual = mxfp4_attention_ref(sampled_query, key, value)
        ones = torch.ones_like(value)
        constant_kernel = _mha_v4_mxfp4(query, key, ones).index_select(1, rows)
        constant_emulator = mxfp4_attention_ref(sampled_query, key, ones)

    torch.testing.assert_close(constant_kernel, torch.ones_like(expected), rtol=0, atol=0)
    torch.testing.assert_close(constant_emulator, torch.ones_like(actual), rtol=0, atol=0)
    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    relative_rmse = (
        (actual_f32 - expected_f32).square().mean().sqrt()
        / expected_f32.square().mean().sqrt().clamp_min(1.0e-30)
    ).item()
    cosine = F.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    ).item()

    # Bound MFMA accumulation-order residual, not bit-exact agreement.
    assert cosine >= 0.9999, f"S={sequence}: cosine={cosine}, relative_rmse={relative_rmse}"
    assert relative_rmse <= 0.015, f"S={sequence}: cosine={cosine}, relative_rmse={relative_rmse}"


def test_mxfp4_v_tail_matches_sdpa_at_production_remainder():
    sequence = 208
    assert sequence % 128 == 75_600 % 128 == 80
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    shape = (1, sequence, 2, 128)
    query = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    key = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    value = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )

    with torch.no_grad():
        output = _mha_v4_mxfp4(query, key, value)
        reference = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)

    cosine = F.cosine_similarity(
        output.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.95


@pytest.mark.parametrize("sequence", [384, 4096])
def test_repeated_probability_overflow_rollback_matches_kernel(sequence):
    query = torch.ones((1, sequence, 2, 128), dtype=torch.bfloat16, device="cuda")
    key = torch.zeros_like(query)
    key[:, 128] = 4
    key[:, 256] = 8
    generator = torch.Generator(device="cuda").manual_seed(20261029)
    value = torch.randn(query.shape, generator=generator, device="cuda").bfloat16()
    rows = torch.tensor([0, sequence - 1], device="cuda")
    with torch.no_grad():
        kernel = _mha_v4_mxfp4(query, key, value).index_select(1, rows)
        emulator = mxfp4_attention_ref(query.index_select(1, rows), key, value)
        ones = torch.ones_like(value)
        constant_kernel = _mha_v4_mxfp4(query, key, ones).index_select(1, rows)
        constant_emulator = mxfp4_attention_ref(query.index_select(1, rows), key, ones)
    assert torch.isfinite(kernel).all()
    assert torch.isfinite(emulator).all()
    # Rollback can leave tiny prior-tile contributions at channels whose
    # dominant quantized V is zero. MFMA accumulation is not bit-exact to
    # einsum (observed absolute residual <= 4.24e-22 for this fixture).
    torch.testing.assert_close(emulator, kernel, rtol=0.001, atol=1e-6)
    torch.testing.assert_close(constant_kernel, torch.ones_like(kernel), rtol=0, atol=0)
    torch.testing.assert_close(constant_emulator, torch.ones_like(emulator), rtol=0, atol=0)
