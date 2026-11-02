"""Loading GPTQ (on-grid MXFP4) weights into A4W4/A6W4 kernels must be lossless."""
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


@pytest.mark.parametrize("gemm", [KernelGemm, A6W4Gemm])
def test_kernel_weight_packer_equals_reference_rtn_and_is_idempotent(gemm):
    weight, on_grid, x = _weights()
    torch.testing.assert_close(gemm(on_grid)(x), gemm(weight)(x), atol=0, rtol=0)


def test_a6w4_beats_a4w4_on_identical_weights():
    weight, on_grid, x = _weights()
    exact = x.float() @ on_grid.float().T
    error = lambda y: float((y.float() - exact).square().sum() / exact.square().sum())
    assert error(A6W4Gemm(on_grid)(x)) < error(KernelGemm(on_grid)(x))


ARTIFACT = Path(os.environ.get("PTQ_ARTIFACT", "/data/ptq/artifacts/20261026T164220Z-b07b3192c31a"))


@pytest.mark.skipif(not ARTIFACT.exists(), reason="authoritative GPTQ artifact not mounted")
def test_gptq_artifact_requantizes_to_its_own_codes():
    from xfuser.model_executor.quant.ptq.selective_precision import artifact_weight_source
    from safetensors import safe_open
    import json

    config = json.loads((ARTIFACT / "quant_config.json").read_text())
    source = artifact_weight_source(ARTIFACT, "transformer_2", "cuda")
    for name in ("blocks.0.attn1.to_q", "blocks.20.ffn.net.0.proj", "blocks.39.ffn.net.2"):
        entry = config["layers"][f"transformer_2.{name}"]
        with safe_open(ARTIFACT / entry["shard"], framework="pt", device="cpu") as handle:
            packed, scale = handle.get_tensor(entry["packed_key"]), handle.get_tensor(entry["scale_key"])
        repacked, rescale = quantize_mxfp4(source(f"{name}.weight"))
        assert torch.equal(repacked.cpu(), packed) and torch.equal(rescale.cpu(), scale)
