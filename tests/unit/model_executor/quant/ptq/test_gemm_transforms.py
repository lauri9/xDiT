import pytest
import torch

from xfuser.model_executor.quant.ptq.gemm_transforms import (
    activation_compensating_target, block_hadamard, cfg_combined_hessian,
    fold_value_output, mix_cfg_hessian,
    predicted_activation_recovery,
)
from xfuser.model_executor.quant.ptq.mxfp4_ref import (
    dequantize_mxfp4, quantize_mxfp4, quantize_mxfp4_scale_search, quantize_mxfp4_nearest,
)
from xfuser.model_executor.quant.ptq.gptq_mxfp4 import GPTQConfig, gptq_quantize_mxfp4


@pytest.mark.parametrize("size", [32,128])
def test_rotation_preserves_linear_function_and_is_involution(size):
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(7,256,generator=generator)
    weight = torch.randn(11,256,generator=generator)
    torch.testing.assert_close(block_hadamard(block_hadamard(values,size),size), values)
    torch.testing.assert_close(block_hadamard(values,size) @ block_hadamard(weight,size).T, values @ weight.T, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("size", [32,128])
def test_value_fold_preserves_multihead_attention_and_bias(size):
    generator = torch.Generator().manual_seed(12)
    values = torch.randn(2,9,64,generator=generator)
    vw = torch.randn(256,64,generator=generator)
    vb = torch.randn(256,generator=generator)
    ow = torch.randn(17,256,generator=generator)
    ob = torch.randn(17,generator=generator)
    probabilities = torch.randn(2,2,5,9,generator=generator).softmax(-1)
    def attention(vw,vb,ow):
        v = (values @ vw.T+vb).reshape(2,9,2,128).transpose(1,2)
        output = (probabilities @ v).transpose(1,2).reshape(2,5,256)
        return output @ ow.T+ob
    transformed = fold_value_output(vw,vb,ow,size=size)
    torch.testing.assert_close(attention(*transformed),attention(vw,vb,ow),rtol=2e-5,atol=5e-5)


def test_activation_target_recovers_known_input_correlated_error():
    generator = torch.Generator().manual_seed(3)
    values = torch.randn(512,32,generator=generator)
    quantized = values*0.8
    weight = torch.randn(5,32,generator=generator)
    target,damping = activation_compensating_target(weight,quantized.T @ quantized,quantized.T @ values,damping=0)
    assert damping == 0
    torch.testing.assert_close(quantized @ target.T,values @ weight.T,rtol=1e-4,atol=2e-5)


def test_cfg_hessian_uses_paired_rows_and_trace_normalized_mix():
    conditional = torch.tensor([[1.,2.],[3.,-1.]])
    unconditional = torch.tensor([[0.,1.],[2.,3.]])
    combined = cfg_combined_hessian(conditional,unconditional,guidance=3)
    expected = torch.tensor([[3.,4.],[5.,-9.]])
    torch.testing.assert_close(combined,expected.T @ expected)
    pooled = conditional.T @ conditional+unconditional.T @ unconditional
    mixed = mix_cfg_hessian(pooled,combined,fraction=0.5)
    assert float(mixed.trace()) == pytest.approx(1)
    assert torch.linalg.eigvalsh(mixed).min() > 0


def test_scale_search_cannot_worsen_its_weighted_diagonal_objective():
    generator = torch.Generator().manual_seed(19)
    values = torch.randn(12,64,generator=generator)
    diagonal = torch.rand(64,generator=generator)
    p,s = quantize_mxfp4(values)
    q,t = quantize_mxfp4_scale_search(values,diagonal)
    baseline = ((values-dequantize_mxfp4(p,s)).square()*diagonal).reshape(12,2,32).sum(-1)
    searched = ((values-dequantize_mxfp4(q,t)).square()*diagonal).reshape(12,2,32).sum(-1)
    assert (searched <= baseline+1e-6).all()
    assert (t <= s).all() and ((s.to(torch.int16)-t.to(torch.int16)) <= 1).all()


def test_scale_search_inside_gptq_matches_rtn_search_for_identity_hessian():
    values = torch.randn(12,64,generator=torch.Generator().manual_seed(19))
    p,s = quantize_mxfp4_scale_search(values)
    result = gptq_quantize_mxfp4(values,torch.eye(64),GPTQConfig(scale_search=True))
    torch.testing.assert_close(result.packed,p,rtol=0,atol=0)
    torch.testing.assert_close(result.scale,s,rtol=0,atol=0)


def test_nearest_scale_intentionally_saturates_outlier_instead_of_ceil_grid():
    values = torch.ones(1,32)
    values[0,0] = 3.1
    full_p,full_s = quantize_mxfp4(values)
    near_p,near_s = quantize_mxfp4_nearest(values)
    assert int(full_s[0,0]) == 127
    assert int(near_s[0,0]) == 126
    assert float(dequantize_mxfp4(near_p,near_s)[0,0]) == 3.0
    assert float(dequantize_mxfp4(full_p,full_s)[0,0]) == 3.0


def test_precheck_credits_only_activation_recovery_not_removing_weight_error():
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(512,32,generator=generator)
    quantized = values*0.8
    weight = torch.randn(5,32,generator=generator)
    activation_error = float(((values-quantized) @ weight.T).square().sum())
    result = predicted_activation_recovery(weight,quantized.T @ quantized,quantized.T @ values,
        activation_error2=activation_error,total_error2=activation_error*10)
    assert result["predicted_total_error_reduction"] == pytest.approx(0.1,rel=0.005)
    assert result["predicted_float_target_activation_error2"] < activation_error*0.005
