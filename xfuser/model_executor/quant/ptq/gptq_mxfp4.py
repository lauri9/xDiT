"""Offline GPTQ rounding for per-row, per-32 MXFP4 weights."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .mxfp4_ref import dequantize_mxfp4, quantize_mxfp4


@dataclass(frozen=True)
class GPTQConfig:
    percdamp: float = 0.01
    max_cholesky_retries: int = 3


@dataclass
class GPTQResult:
    packed: torch.Tensor
    scale: torch.Tensor
    report: dict

    def dequantize(self) -> torch.Tensor:
        return dequantize_mxfp4(self.packed, self.scale)


@dataclass
class PreparedHessian:
    raw: torch.Tensor
    inverse_cholesky: torch.Tensor
    dead: torch.Tensor
    damping: float
    retries: int


def proxy_loss(weight: torch.Tensor, quantized: torch.Tensor, hessian: torch.Tensor) -> float:
    error = weight.to(torch.float32) - quantized.to(torch.float32)
    hessian_f32 = hessian.to(device=error.device, dtype=torch.float32)
    return float(((error @ hessian_f32) * error).sum().item())


def _inverse_cholesky(
    hessian: torch.Tensor,
    config: GPTQConfig,
) -> tuple[torch.Tensor, float, int]:
    diagonal_mean = hessian.diag().mean()
    base_damping = float(config.percdamp * diagonal_mean)
    if base_damping <= 0:
        base_damping = config.percdamp or 1.0e-6

    last_error = None
    for retry in range(config.max_cholesky_retries + 1):
        damping = base_damping * (10**retry)
        damped = hessian.clone()
        damped.diagonal().add_(damping)
        try:
            lower = torch.linalg.cholesky(damped)
            inverse = torch.cholesky_inverse(lower)
            return torch.linalg.cholesky(inverse, upper=True), damping, retry
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(
        f"GPTQ Cholesky failed after {config.max_cholesky_retries + 1} attempts"
    ) from last_error


def prepare_hessian(
    hessian: torch.Tensor,
    config: GPTQConfig = GPTQConfig(),
) -> PreparedHessian:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Hessian must be square")
    raw = hessian.to(torch.float32)
    work = raw.clone()
    dead = work.diag() == 0
    if dead.any():
        work[dead, :] = 0
        work[:, dead] = 0
        work[dead, dead] = 1
    inverse_cholesky, damping, retries = _inverse_cholesky(work, config)
    return PreparedHessian(
        raw=raw,
        inverse_cholesky=inverse_cholesky,
        dead=dead,
        damping=damping,
        retries=retries,
    )


@torch.no_grad()
def gptq_quantize_mxfp4(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    config: GPTQConfig = GPTQConfig(),
    *,
    prepared: PreparedHessian | None = None,
) -> GPTQResult:
    """Quantize one ``[N, K]`` weight using a saved ``[K, K]`` Hessian."""

    if weight.ndim != 2 or hessian.ndim != 2:
        raise ValueError("GPTQ weight and Hessian must be two-dimensional")
    out_features, in_features = weight.shape
    if in_features % 32:
        raise ValueError("GPTQ MXFP4 requires K divisible by 32")
    if tuple(hessian.shape) != (in_features, in_features):
        raise ValueError("Hessian shape does not match the weight K dimension")

    device = weight.device
    original = weight.to(torch.float32)
    prepared = prepared or prepare_hessian(
        hessian.to(device=device, dtype=torch.float32),
        config,
    )
    if prepared.raw.device != device:
        raise ValueError("prepared Hessian must be on the weight device")
    if tuple(prepared.raw.shape) != (in_features, in_features):
        raise ValueError("prepared Hessian shape does not match the weight")
    dead = prepared.dead
    inverse_cholesky = prepared.inverse_cholesky
    rtn_packed, rtn_scale = quantize_mxfp4(original)
    rtn = dequantize_mxfp4(rtn_packed, rtn_scale)

    work = original.clone()
    packed = torch.empty(
        (out_features, in_features // 2),
        dtype=torch.uint8,
        device=device,
    )
    scales = torch.empty(
        (out_features, in_features // 32),
        dtype=torch.uint8,
        device=device,
    )

    for start in range(0, in_features, 32):
        end = start + 32
        if dead[start:end].any():
            # One E8M0 scale is shared by the 32 columns. Keeping a dead
            # column exactly at RTN therefore pins the whole group's grid.
            group_packed = rtn_packed[:, start // 2 : end // 2]
            group_scale = rtn_scale[:, start // 32 : end // 32]
            group_quantized = rtn[:, start:end].clone()
        else:
            group_packed, group_scale = quantize_mxfp4(work[:, start:end])
            group_quantized = dequantize_mxfp4(group_packed, group_scale)
        group_errors = torch.zeros(
            (out_features, 32),
            dtype=torch.float32,
            device=device,
        )

        for local_index, column in enumerate(range(start, end)):
            if dead[column]:
                group_quantized[:, local_index] = rtn[:, column]
                continue
            diagonal = inverse_cholesky[column, column]
            error = (work[:, column] - group_quantized[:, local_index]) / diagonal
            group_errors[:, local_index] = error
            work[:, column:end].add_(
                error.unsqueeze(1) @ inverse_cholesky[column, column:end].unsqueeze(0),
                alpha=-1,
            )

        packed[:, start // 2 : end // 2] = group_packed
        scales[:, start // 32] = group_scale[:, 0]
        if end < in_features:
            work[:, end:].add_(
                group_errors @ inverse_cholesky[start:end, end:],
                alpha=-1,
            )

    quantized = dequantize_mxfp4(packed, scales)
    raw_hessian = prepared.raw
    rtn_loss = proxy_loss(original, rtn, raw_hessian)
    gptq_loss = proxy_loss(original, quantized, raw_hessian)
    return GPTQResult(
        packed=packed,
        scale=scales,
        report={
            "damping": prepared.damping,
            "cholesky_retries": prepared.retries,
            "dead_columns": int(dead.sum().item()),
            "rtn_proxy_loss": rtn_loss,
            "gptq_proxy_loss": gptq_loss,
            "relative_improvement": (
                (rtn_loss - gptq_loss) / rtn_loss if rtn_loss else 0.0
            ),
        },
    )
