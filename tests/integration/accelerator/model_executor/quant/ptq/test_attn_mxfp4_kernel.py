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

    # exp2/online-state approximation: bound residual, not bit-exact agreement.
    assert cosine >= 0.998, f"S={sequence}: cosine={cosine}, relative_rmse={relative_rmse}"
    assert relative_rmse <= 0.065, f"S={sequence}: cosine={cosine}, relative_rmse={relative_rmse}"


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
