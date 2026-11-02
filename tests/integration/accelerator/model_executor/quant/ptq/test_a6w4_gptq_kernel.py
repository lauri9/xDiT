"""Weight paths for the A4W4/A6W4 GPTQ study must be lossless.

A4W4: the per_1x32 packer equals reference RTN and keeps on-grid (GPTQ)
weights. A6W4: AITER rotates K by a BF16 H32 before quantizing, so RTN and
GPTQ live in the rotated domain; codes packed directly must equal the packer.
"""
import os
from pathlib import Path

import pytest
import torch

from xfuser.model_executor.quant.ptq.block_sim import A6W4Gemm, KernelGemm
from xfuser.model_executor.quant.ptq.mxfp4_ref import dequantize_mxfp4, quantize_mxfp4

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


def _weights():
    torch.manual_seed(3)
    weight = torch.randn(512, 1024, device="cuda") / 30
    weight[:, ::97] *= 40  # outlier columns stress the per-32 scales
    weight = weight.bfloat16()
    on_grid = dequantize_mxfp4(*quantize_mxfp4(weight)).bfloat16()
    x = (torch.randn(257, 1024, device="cuda") * torch.linspace(0.1, 6, 1024, device="cuda")).bfloat16()
    return weight, on_grid, x


def test_a4w4_packer_equals_reference_rtn_and_is_idempotent():
    weight, on_grid, x = _weights()
    torch.testing.assert_close(KernelGemm(on_grid)(x), KernelGemm(weight)(x), atol=0, rtol=0)


def test_a6w4_direct_codes_equal_aiter_packer():
    from xfuser.model_executor.quant.ptq.a6w4_weights import rotated_weight

    weight, _, x = _weights()
    codes = quantize_mxfp4(rotated_weight(weight))
    torch.testing.assert_close(A6W4Gemm(None, codes=codes)(x), A6W4Gemm(weight)(x), atol=0, rtol=0)


def test_a6w4_identity_hessian_gptq_reproduces_rtn_and_conjugation_is_consistent():
    from xfuser.model_executor.quant.ptq.a6w4_weights import conjugate_hessian, kernel_h32, rotated_weight
    from xfuser.model_executor.quant.ptq.gptq_mxfp4 import gptq_quantize_mxfp4

    weight, _, x = _weights()
    h32 = kernel_h32("cuda")
    rotated = rotated_weight(weight)
    result = gptq_quantize_mxfp4(rotated, torch.eye(1024, device="cuda"))
    reference = quantize_mxfp4(rotated)
    assert torch.equal(result.packed, reference[0]) and torch.equal(result.scale, reference[1])
    activations = x.float()
    rotated_x = (activations.reshape(-1, 32, 32) @ h32).reshape(activations.shape)
    torch.testing.assert_close(conjugate_hessian(activations.T @ activations, h32), rotated_x.T @ rotated_x,
                               rtol=1e-4, atol=1e-2)


def test_a6w4_beats_a4w4_on_identical_weights():
    weight, on_grid, x = _weights()
    exact = x.float() @ on_grid.float().T
    error = lambda y: float((y.float() - exact).square().sum() / exact.square().sum())
    assert error(A6W4Gemm(on_grid)(x)) < error(KernelGemm(on_grid)(x))


ARTIFACT = Path(os.environ.get("PTQ_ARTIFACT", "/data/ptq/artifacts/20261026T164220Z-b07b3192c31a"))


@pytest.mark.skipif(not ARTIFACT.exists(), reason="authoritative GPTQ artifact not mounted")
def test_gptq_artifact_values_survive_a4w4_repacking():
    """GPTQ may keep a larger scale than its final group needs; re-quantizing
    then picks the smaller exact scale. Codes change, values must not."""
    from xfuser.model_executor.quant.ptq.selective_precision import artifact_weight_source

    source = artifact_weight_source(ARTIFACT, "transformer_2", "cuda")
    for name in ("blocks.0.attn1.to_q", "blocks.20.ffn.net.0.proj", "blocks.39.ffn.net.2"):
        weight = source(f"{name}.weight")
        assert torch.equal(dequantize_mxfp4(*quantize_mxfp4(weight)), weight.float())
        # One-hot activations are exact in MXFP4, so GEMM(I) returns the stored weight.
        stored = KernelGemm(weight)(torch.eye(weight.shape[1], device="cuda", dtype=torch.bfloat16))
        assert torch.equal(stored.float().T, weight.float())
