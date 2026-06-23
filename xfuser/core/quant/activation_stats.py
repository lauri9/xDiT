import csv
import logging
import os
from typing import Optional

import torch

from xfuser.envs import environment_variables, parse_env_bool

logger = logging.getLogger(__name__)

_layer_call_counts: dict[str, int] = {}
_logged_layers: set[str] = set()
_csv_header_written = False


def should_log_activation_stats() -> bool:
    return parse_env_bool(environment_variables["MXFP4_LOG_ACTIVATION_STATS"](), default=False)


def _should_log_on_this_rank() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def _layer_filter_matches(layer_name: str) -> bool:
    raw = environment_variables["MXFP4_LOG_ACTIVATION_STATS_LAYERS"]()
    if not raw:
        return True
    needles = [part.strip() for part in str(raw).split(",") if part.strip()]
    return any(needle in layer_name for needle in needles)


def compute_mx_group_stats(x: torch.Tensor, group_size: int = 32) -> dict[str, float]:
    """Compute MX-group outlier concentration metrics on [M, K] activations."""
    if x.shape[-1] % group_size != 0:
        raise ValueError(
            f"Last dimension ({x.shape[-1]}) must be divisible by group_size ({group_size})"
        )
    with torch.no_grad():
        x = x.detach().float()
        num_groups = x.shape[-1] // group_size
        grouped = x.view(*x.shape[:-1], num_groups, group_size)
        abs_grouped = grouped.abs()
        rms = grouped.pow(2).mean(dim=-1).sqrt().clamp(min=1e-12)
        max_abs = abs_grouped.max(dim=-1).values
        ratio = (max_abs / rms).reshape(-1)

        flat_abs = x.abs().reshape(-1)
        p50 = torch.quantile(flat_abs, 0.5).item()
        p99 = torch.quantile(flat_abs, 0.99).item()
        per_group_max = max_abs.reshape(-1)
        group_mean = per_group_max.mean().item()
        group_std = per_group_max.std(unbiased=False).item()

        return {
            "group_max_over_rms_mean": ratio.mean().item(),
            "group_max_over_rms_p99": torch.quantile(ratio, 0.99).item(),
            "global_p99_over_p50": p99 / max(p50, 1e-12),
            "group_scale_spread": group_std / max(group_mean, 1e-12),
            "max_abs": flat_abs.max().item(),
            "rms": x.pow(2).mean().sqrt().item(),
        }


def _sample_rows(x: torch.Tensor, sample_tokens: int) -> torch.Tensor:
    if x.shape[0] <= sample_tokens:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:sample_tokens]
    return x.index_select(0, idx)


def _write_csv_row(path: str, row: dict[str, object]) -> None:
    global _csv_header_written
    write_header = not _csv_header_written and not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
            _csv_header_written = True
        writer.writerow(row)


def maybe_log_activation_stats(
    *,
    layer_name: str,
    x: torch.Tensor,
    group_size: int,
    phase: str,
    hadamard_enabled: bool,
    step: Optional[int] = None,
) -> None:
    """Rate-limited activation distribution logging for MXFP4 GEMM diagnostics."""
    if not should_log_activation_stats() or not _should_log_on_this_rank():
        return
    if not layer_name or not _layer_filter_matches(layer_name):
        return

    max_layers = int(environment_variables["MXFP4_LOG_ACTIVATION_STATS_MAX_LAYERS"]())
    if layer_name not in _logged_layers and len(_logged_layers) >= max_layers:
        return

    every_n = max(1, int(environment_variables["MXFP4_LOG_ACTIVATION_STATS_EVERY"]()))
    _layer_call_counts[layer_name] = _layer_call_counts.get(layer_name, 0) + 1
    if (_layer_call_counts[layer_name] - 1) % every_n != 0:
        return

    _logged_layers.add(layer_name)
    sample_tokens = int(environment_variables["MXFP4_LOG_ACTIVATION_STATS_SAMPLE_TOKENS"]())
    x_2d = x.reshape(-1, x.shape[-1])
    x_sample = _sample_rows(x_2d, sample_tokens).cpu()
    stats = compute_mx_group_stats(x_sample, group_size=group_size)
    row = {
        "step": step if step is not None else "",
        "layer_name": layer_name,
        "phase": phase,
        "hadamard_enabled": int(hadamard_enabled),
        "group_size": group_size,
        **stats,
    }

    output_path = environment_variables["MXFP4_LOG_ACTIVATION_STATS_OUTPUT"]()
    if output_path:
        _write_csv_row(output_path, row)
    else:
        logger.info(
            "MXFP4 activation stats layer=%s phase=%s hadamard=%s %s",
            layer_name,
            phase,
            hadamard_enabled,
            " ".join(f"{k}={v:.6g}" for k, v in stats.items()),
        )
