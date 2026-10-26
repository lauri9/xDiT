import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.gptq_mxfp4 import (
    GPTQConfig,
    gptq_quantize_mxfp4,
)
from xfuser.model_executor.quant.ptq.mxfp4_ref import quantize_mxfp4


def _weight():
    return torch.randn(8, 64, generator=torch.Generator().manual_seed(11))


def test_identity_hessian_reproduces_rtn_exactly():
    weight = _weight()
    rtn_packed, rtn_scale = quantize_mxfp4(weight)

    result = gptq_quantize_mxfp4(weight, torch.eye(64))

    torch.testing.assert_close(result.packed, rtn_packed, rtol=0, atol=0)
    torch.testing.assert_close(result.scale, rtn_scale, rtol=0, atol=0)
    assert result.report["gptq_proxy_loss"] == pytest.approx(
        result.report["rtn_proxy_loss"]
    )


def test_gptq_does_not_increase_correlated_proxy_loss():
    generator = torch.Generator().manual_seed(5)
    latent = torch.randn(512, 8, generator=generator)
    mixing = torch.randn(8, 64, generator=generator)
    inputs = latent @ mixing + 0.01 * torch.randn(512, 64, generator=generator)
    hessian = inputs.T @ inputs

    result = gptq_quantize_mxfp4(_weight(), hessian)

    assert result.report["gptq_proxy_loss"] <= result.report["rtn_proxy_loss"] * (
        1 + 1e-5
    )


def test_dead_columns_keep_their_entire_scale_group_at_rtn():
    weight = _weight()
    hessian = torch.eye(64)
    hessian[3, 3] = 0
    rtn_packed, rtn_scale = quantize_mxfp4(weight)

    result = gptq_quantize_mxfp4(weight, hessian)

    torch.testing.assert_close(
        result.packed[:, :16],
        rtn_packed[:, :16],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(result.scale[:, :1], rtn_scale[:, :1], rtol=0, atol=0)
    assert result.report["dead_columns"] == 1


def test_invalid_hessian_shape_is_rejected():
    with pytest.raises(ValueError, match="does not match"):
        gptq_quantize_mxfp4(_weight(), torch.eye(32), GPTQConfig())
