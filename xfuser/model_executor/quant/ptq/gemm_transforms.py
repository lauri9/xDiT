"""Opt-in offline transforms/objectives for the Workstream G experiments."""
from __future__ import annotations

import math

import torch


def block_hadamard(values: torch.Tensor, size: int) -> torch.Tensor:
    """Normalized FWHT within the last-dimension blocks; FP32 arithmetic."""
    if size not in (32, 128) or values.shape[-1] % size:
        raise ValueError("rotation size must be 32 or 128 and divide the last dimension")
    shape = values.shape
    work = values.float().reshape(-1, size)
    stride = 1
    while stride < size:
        pairs = work.reshape(-1, size // (2*stride), 2, stride)
        first, second = pairs[:, :, 0], pairs[:, :, 1]
        work = torch.stack((first+second, first-second), dim=2).reshape(-1,size)
        stride *= 2
    return (work / math.sqrt(size)).reshape(shape)


def fold_value_output(value_weight, value_bias, output_weight, *, size):
    """Wv'=R^T Wv, bv'=bv R, Wo'=Wo R, within each 128-wide head.

    Returned tensors are FP32. Quantization/storage decisions belong to
    the experiment driver, not this function-preserving algebra.
    """
    if value_weight.shape[0] % 128 or output_weight.shape[1] != value_weight.shape[0]:
        raise ValueError("fold requires matching 128-wide attention heads")
    return (block_hadamard(value_weight.T, size).T.contiguous(),
        None if value_bias is None else block_hadamard(value_bias, size),
        block_hadamard(output_weight, size))


def activation_compensating_target(weight, quantized_hessian, cross_moment, *, damping=0.01):
    """W C^T (Hq + damp I)^-1, with C=Qa(X)^T X.

    The damping regularizes only ill-constrained activation directions.
    Returns the target and absolute diagonal damping for provenance.
    """
    width = weight.shape[1]
    if quantized_hessian.shape != (width,width) or cross_moment.shape != (width,width):
        raise ValueError("Hq and C must match weight input dimensions")
    if damping < 0:
        raise ValueError("damping must be nonnegative")
    hessian = quantized_hessian.float().clone()
    amount = float(hessian.diag().mean())*damping
    hessian.diagonal().add_(amount)
    target = torch.linalg.solve(hessian, cross_moment.float() @ weight.float().T).T
    return target, amount


def cfg_combined_hessian(conditional, unconditional, *, guidance):
    if conditional.shape != unconditional.shape or conditional.ndim != 2:
        raise ValueError("CFG rows must be paired [tokens, channels]")
    if guidance <= 1:
        raise ValueError("guided CFG requires guidance above one")
    values = guidance*conditional.float()-(guidance-1)*unconditional.float()
    return values.T @ values


def mix_cfg_hessian(pooled, combined, *, fraction):
    if pooled.shape != combined.shape or pooled.ndim != 2 or pooled.shape[0] != pooled.shape[1]:
        raise ValueError("CFG Hessians must be matching square matrices")
    if fraction not in (0.5,1.0):
        raise ValueError("pre-registered CFG fractions are 0.5 and 1")
    if pooled.trace() <= 0 or combined.trace() <= 0:
        raise ValueError("CFG Hessians require positive trace")
    return (1-fraction)*pooled.float()/pooled.trace()+fraction*combined.float()/combined.trace()
