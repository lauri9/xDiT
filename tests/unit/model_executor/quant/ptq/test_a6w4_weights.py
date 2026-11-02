import math

import torch

from xfuser.model_executor.quant.ptq.a6w4_weights import conjugate_hessian, effective_weight
from xfuser.model_executor.quant.ptq.mxfp4_ref import dequantize_mxfp4, quantize_mxfp4
from xfuser.model_executor.quant.ptq.rotations import sylvester_hadamard

H32 = (sylvester_hadamard(32) / math.sqrt(32)).float()


def test_conjugated_hessian_is_the_rotated_activation_moment():
    x = torch.randn(200, 96, dtype=torch.float64).float()
    rotated = (x.reshape(200, 3, 32) @ H32).reshape(200, 96)
    torch.testing.assert_close(conjugate_hessian(x.T @ x, H32), rotated.T @ rotated, rtol=1e-4, atol=1e-3)


def test_effective_weight_undoes_the_rotation_of_quantized_codes():
    weight = torch.randn(8, 64)
    rotated = (weight.reshape(8, 2, 32) @ H32).reshape(8, 64)
    codes = quantize_mxfp4(rotated)
    effective = effective_weight(*codes, H32)
    x = torch.randn(5, 64)
    x_rot = (x.reshape(5, 2, 32) @ H32).reshape(5, 64)
    torch.testing.assert_close(x @ effective.T, x_rot @ dequantize_mxfp4(*codes).T, rtol=1e-4, atol=1e-4)
