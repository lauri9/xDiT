"""Eager-BF16 Wan 2.2 calibration entry point using AITER MHA v4."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from .observers import WanHessianObservers


MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


@dataclass(frozen=True)
class CalibrationSettings:
    prompts_path: Path
    output_root: Path = Path("/data/ptq/calib")
    model: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    num_inference_steps: int = 20
    guidance_scale: float = 4.0
    guidance_scale_2: float = 3.0
    flow_shift: float = 3.0
    boundary_ratio: float = 0.875
    base_seed: int = 42
    attention_backend: str = "AITER_BF16"
    cross_attention_backend: str = "AITER_BF16"
    token_sample_rate: float = 1.0
    tokens_per_forward: int | None = None
    max_prompts: int | None = None
    block_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.model_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.model_revision.lower()
        ):
            raise ValueError("calibration requires an exact 40-character source commit")
        if self.guidance_scale <= 1 or self.guidance_scale_2 <= 1:
            raise ValueError("calibration requires both CFG branches")
        if self.boundary_ratio != 0.875:
            raise ValueError("unexpected Wan 2.2 expert boundary")
        if not 0 < self.token_sample_rate <= 1:
            raise ValueError("token_sample_rate must be in (0, 1]")
        if self.block_indices is not None and (
            len(set(self.block_indices)) != len(self.block_indices)
            or any(index < 0 for index in self.block_indices)
        ):
            raise ValueError("block_indices must be unique non-negative integers")


def prompt_seed(settings: CalibrationSettings, global_index: int) -> int:
    return settings.base_seed + global_index


def prompt_indices(total: int, rank: int, world_size: int) -> tuple[int, ...]:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid data-parallel rank")
    return tuple(range(rank, total, world_size))


def read_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("calibration prompt file contains an empty prompt")
    return prompts


def assert_no_prompt_overlap(calibration: list[str], evaluation: list[str]) -> None:
    overlap = set(calibration) & set(evaluation)
    if overlap:
        raise ValueError(f"calibration/evaluation prompt overlap: {sorted(overlap)[:3]}")


def _settings_payload(settings: CalibrationSettings) -> dict:
    payload = dataclasses.asdict(settings)
    payload["prompts_path"] = str(settings.prompts_path)
    payload["output_root"] = str(settings.output_root)
    return payload


def _calib_id(settings: CalibrationSettings, prompt_sha256: str) -> str:
    payload = {**_settings_payload(settings), "prompt_sha256": prompt_sha256}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{digest}"


def build_eager_bf16_args(
    settings: CalibrationSettings,
    world_size: int,
    *,
    gemm_quantization: str | None = None,
    ptq_artifact_path: str | None = None,
):
    from xfuser import xFuserArgs

    return xFuserArgs(
        model=settings.model,
        model_revision=settings.model_revision,
        height=settings.height,
        width=settings.width,
        num_frames=settings.num_frames,
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        guidance_scale_2=settings.guidance_scale_2,
        flow_shift=settings.flow_shift,
        seed=settings.base_seed,
        output_type="latent",
        attention_backend=settings.attention_backend,
        cross_attention_backend=settings.cross_attention_backend,
        data_parallel_degree=world_size,
        ulysses_degree=1,
        ring_degree=1,
        tensor_parallel_degree=1,
        pipefusion_parallel_degree=1,
        fully_shard_degree=1,
        use_cfg_parallel=False,
        use_torch_compile=False,
        use_fp4_gemms=False,
        use_fp8_gemms=False,
        use_int8_gemms=False,
        gemm_quantization=gemm_quantization,
        ptq_artifact_path=ptq_artifact_path,
        use_fp8_text_encoder=False,
        use_fp8_comms=False,
        use_hybrid_attn_schedule=False,
        use_hybrid_gemm_schedule=False,
        cache_method=None,
        memory_efficient_sharding=False,
        memory_efficient_replicated_load=False,
    )


def preflight_eager_bf16(model, settings: CalibrationSettings) -> None:
    config = model.config
    disabled = {
        "use_torch_compile": config.use_torch_compile,
        "use_fp4_gemms": config.use_fp4_gemms,
        "use_fp8_gemms": config.use_fp8_gemms,
        "use_int8_gemms": config.use_int8_gemms,
        "use_fp8_text_encoder": config.use_fp8_text_encoder,
        "use_fp8_comms": config.use_fp8_comms,
        "use_hybrid_attn_schedule": config.use_hybrid_attn_schedule,
        "use_hybrid_gemm_schedule": config.use_hybrid_gemm_schedule,
        "cache_method": config.cache_method,
    }
    active = [name for name, value in disabled.items() if value]
    if active:
        raise RuntimeError(f"calibration optimization unexpectedly active: {active}")
    degrees = {
        "ulysses": config.ulysses_degree,
        "ring": config.ring_degree,
        "tensor": config.tensor_parallel_degree,
        "pipefusion": config.pipefusion_parallel_degree,
        "fsdp": config.fully_shard_degree,
    }
    if any(value != 1 for value in degrees.values()):
        raise RuntimeError(f"calibration has non-DP parallelism: {degrees}")
    if config.model_revision != settings.model_revision:
        raise RuntimeError("calibration source revision changed during setup")
    attention = {
        "self": config.attention_backend,
        "cross": config.cross_attention_backend,
    }
    expected_attention = {
        "self": settings.attention_backend,
        "cross": settings.cross_attention_backend,
    }
    if attention != expected_attention:
        raise RuntimeError(
            f"calibration attention backend changed during setup: {attention}"
        )
    if not hasattr(model.pipe, "transformer") or not hasattr(model.pipe, "transformer_2"):
        raise RuntimeError("Wan 2.2 calibration requires both denoising experts")


def _all_reduce_observers(observers: WanHessianObservers) -> None:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return
    for accumulator in observers.groups.values():
        torch.distributed.all_reduce(accumulator.hessian, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(accumulator.sum_squares, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(accumulator.absmax, op=torch.distributed.ReduceOp.MAX)
        count = torch.tensor(
            [accumulator.token_count],
            dtype=torch.int64,
            device=accumulator.device,
        )
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        accumulator.token_count = int(count.item())


def _save_observers(
    output_dir: Path,
    expert_name: str,
    observers: WanHessianObservers,
) -> None:
    state = {}
    stats = {}
    for name, accumulator in observers.groups.items():
        state[name] = accumulator.hessian.cpu()
        stats[name] = {
            "token_count": accumulator.token_count,
            "absmax": accumulator.absmax.cpu(),
            "rms": accumulator.rms.cpu(),
        }
    torch.save(state, output_dir / f"{expert_name}_hessians.pt")
    torch.save(stats, output_dir / f"{expert_name}_stats.pt")


def _write_hessian_stats(
    output_dir: Path,
    expert_observers: dict[str, WanHessianObservers],
) -> None:
    with (output_dir / "hessian_stats.jsonl").open("w", encoding="utf-8") as handle:
        for expert_name, observers in expert_observers.items():
            for name, accumulator in observers.groups.items():
                diagonal = accumulator.hessian.diagonal().float().cpu()
                top_count = min(16, diagonal.numel())
                top_values, top_indices = torch.topk(diagonal, top_count)
                positive = diagonal[diagonal > 0]
                entry = {
                    "expert": expert_name,
                    "group": name,
                    "token_count": accumulator.token_count,
                    "trace": diagonal.sum().item(),
                    "diagonal_min": diagonal.min().item(),
                    "diagonal_mean": diagonal.mean().item(),
                    "diagonal_max": diagonal.max().item(),
                    "dead_channels": int((diagonal == 0).sum().item()),
                    "diagonal_condition": (
                        diagonal.max().item() / positive.min().item()
                        if positive.numel()
                        else None
                    ),
                    "top_channels": [
                        {"index": int(index), "diagonal": float(value)}
                        for index, value in zip(top_indices, top_values)
                    ],
                }
                handle.write(json.dumps(entry, sort_keys=True) + "\n")


def run_calibration(settings: CalibrationSettings) -> Path:
    from xfuser.core.distributed import init_distributed_environment

    if not torch.distributed.is_initialized():
        init_distributed_environment()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    prompts = read_prompts(settings.prompts_path)
    if settings.max_prompts is not None:
        prompts = prompts[: settings.max_prompts]
    prompt_bytes = settings.prompts_path.read_bytes()
    prompt_sha256 = hashlib.sha256(prompt_bytes).hexdigest()
    calib_id = _calib_id(settings, prompt_sha256)
    output_dir = settings.output_root / calib_id
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    torch.distributed.barrier()

    args = build_eager_bf16_args(settings, world_size)

    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    model = MODEL_REGISTRY[settings.model](args)
    input_args = {
        "height": settings.height,
        "width": settings.width,
        "num_frames": settings.num_frames,
        "num_inference_steps": settings.num_inference_steps,
        "guidance_scale": settings.guidance_scale,
        "guidance_scale_2": settings.guidance_scale_2,
        "flow_shift": settings.flow_shift,
        "prompt": prompts[0],
        "negative_prompt": model.default_input_values.negative_prompt,
        "seed": settings.base_seed,
    }
    model.initialize(input_args)
    preflight_eager_bf16(model, settings)
    torch.cuda.reset_peak_memory_stats()
    calibration_started_at = time.perf_counter()

    observer_kwargs = {
        "device": f"cuda:{torch.cuda.current_device()}",
        "token_sample_rate": settings.token_sample_rate,
        "tokens_per_forward": settings.tokens_per_forward,
        "block_indices": settings.block_indices,
    }
    high = WanHessianObservers(model.pipe.transformer, **observer_kwargs)
    low = WanHessianObservers(model.pipe.transformer_2, **observer_kwargs)
    try:
        local_indices = prompt_indices(len(prompts), rank, world_size)
        prompt_wall_seconds = {}
        for index in local_indices:
            prompt_started_at = time.perf_counter()
            generator = torch.Generator(
                device=model.pipe._execution_device
            ).manual_seed(prompt_seed(settings, index))
            output = model.pipe(
                height=settings.height,
                width=settings.width,
                num_frames=settings.num_frames,
                num_inference_steps=settings.num_inference_steps,
                prompt=prompts[index],
                negative_prompt=model.default_input_values.negative_prompt,
                guidance_scale=settings.guidance_scale,
                guidance_scale_2=settings.guidance_scale_2,
                generator=generator,
                output_type="latent",
            )
            del output
            torch.cuda.synchronize()
            prompt_wall_seconds[str(index)] = time.perf_counter() - prompt_started_at
    finally:
        high.remove()
        low.remove()

    _all_reduce_observers(high)
    _all_reduce_observers(low)
    if rank == 0:
        _save_observers(output_dir, "transformer", high)
        _save_observers(output_dir, "transformer_2", low)
        _write_hessian_stats(
            output_dir,
            {"transformer": high, "transformer_2": low},
        )
        scheduler = model.pipe.scheduler
        record = {
            "calib_id": calib_id,
            "settings": _settings_payload(settings),
            "prompts": {
                "path": str(settings.prompts_path),
                "sha256": prompt_sha256,
                "count": len(prompts),
                "rank_indices": {
                    str(worker): list(prompt_indices(len(prompts), worker, world_size))
                    for worker in range(world_size)
                },
                "seeds": {
                    str(index): prompt_seed(settings, index)
                    for index in range(len(prompts))
                },
            },
            "negative_prompt_sha256": hashlib.sha256(
                model.default_input_values.negative_prompt.encode()
            ).hexdigest(),
            "scheduler": {
                "timesteps": scheduler.timesteps.detach().cpu().tolist(),
                "sigmas": (
                    scheduler.sigmas.detach().cpu().tolist()
                    if hasattr(scheduler, "sigmas")
                    else None
                ),
            },
            "accumulation": {
                "dtype": "fp32",
                "input_dtype": "bf16",
                "all_reduce": "SUM",
                "prompt_wall_seconds": prompt_wall_seconds,
                "total_wall_seconds": time.perf_counter() - calibration_started_at,
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            },
        }
        (output_dir / "calib_record.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return output_dir


def _parse_args() -> CalibrationSettings:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/data/ptq/calib"))
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--token-sample-rate", type=float, default=1.0)
    parser.add_argument("--tokens-per-forward", type=int)
    parser.add_argument(
        "--blocks",
        help="Comma-separated block indices for a technical spike; default observes all blocks.",
    )
    args = parser.parse_args()
    return CalibrationSettings(
        prompts_path=args.prompts,
        output_root=args.output_root,
        model_revision=args.model_revision,
        max_prompts=args.max_prompts,
        token_sample_rate=args.token_sample_rate,
        tokens_per_forward=args.tokens_per_forward,
        block_indices=(
            tuple(int(value) for value in args.blocks.split(","))
            if args.blocks
            else None
        ),
    )


if __name__ == "__main__":
    print(run_calibration(_parse_args()))
