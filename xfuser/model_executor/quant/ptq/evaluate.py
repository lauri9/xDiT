"""Run and compare Wan 2.2 PTQ accuracy-ladder traces through rung 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .artifact import load_artifact
from .calibrate import (
    CalibrationSettings,
    MODEL_ID,
    MODEL_REVISION,
    assert_no_prompt_overlap,
    build_eager_bf16_args,
    read_prompts,
)
from .mxfp4_ref import dequantize_mxfp4, quantize_mxfp4
from .attn_mxfp4_ref import mxfp4_attention_decomposition
from .attn_transforms import load_attention_artifact
from .provenance import runtime_manifest


REGISTERED_VARIANTS = ("t1", "t2", "t1+t2", "t1+t2+t3")


@lru_cache(maxsize=4)
def _validated_artifact(path: str):
    return load_artifact(path, expected_source_revision=MODEL_REVISION)


@lru_cache(maxsize=8)
def _validated_attention_artifact(path: str):
    return load_attention_artifact(path, expected_source_revision=MODEL_REVISION)


def _tensor_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "sample"):
        return output.sample
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"cannot extract tensor from {type(output).__name__}")


def _sample_rows(tensor: torch.Tensor, count: int = 16) -> torch.Tensor:
    flat = tensor.reshape(-1, tensor.shape[-1])
    if flat.shape[0] <= count:
        return flat.detach().cpu()
    indices = torch.linspace(
        0, flat.shape[0] - 1, count, device=flat.device
    ).round().long()
    return flat.index_select(0, indices).detach().cpu()


@dataclass
class ErrorSums:
    error2: float = 0.0
    signal2: float = 0.0
    dot: float = 0.0
    reference2: float = 0.0
    candidate2: float = 0.0
    values: int = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        reference = reference.float()
        candidate = candidate.float()
        self.error2 += (reference - candidate).square().sum().item()
        self.signal2 += reference.square().sum().item()
        self.dot += (reference * candidate).sum().item()
        self.reference2 += reference.square().sum().item()
        self.candidate2 += candidate.square().sum().item()
        self.values += reference.numel()

    def report(self) -> dict:
        relative_mse = self.error2 / max(self.signal2, 1.0e-30)
        return {
            "relative_mse": relative_mse,
            "sqnr_db": -10.0 * math.log10(max(relative_mse, 1.0e-30)),
            "cosine_similarity": self.dot
            / max(math.sqrt(self.reference2 * self.candidate2), 1.0e-30),
            "values": self.values,
        }


def _artifact_block_weights(
    artifact_path: Path,
    component: str,
    block_index: int,
    device: str,
) -> dict[str, torch.Tensor]:
    artifact = _validated_artifact(str(artifact_path))
    prefix = f"{component}.blocks.{block_index}."
    weights = {}
    for full_name, entry in artifact.config["layers"].items():
        if not full_name.startswith(prefix):
            continue
        with safe_open(
            artifact.root / entry["shard"], framework="pt", device="cpu"
        ) as handle:
            packed = handle.get_tensor(entry["packed_key"])
            scale = handle.get_tensor(entry["scale_key"])
        weights[full_name[len(component) + 1 :]] = dequantize_mxfp4(
            packed, scale
        ).to(device=device, dtype=torch.bfloat16)
    return weights


class LayerErrorHooks:
    """Evaluate block-0 weights on deterministic held-out BF16 activations."""

    def __init__(self, pipe, artifact_path: Path, rows: int = 32):
        self.handles = []
        self.sums: dict[tuple[str, str, str], ErrorSums] = defaultdict(ErrorSums)
        self.calls = defaultdict(int)
        self.rows = rows
        for component in ("transformer", "transformer_2"):
            transformer = getattr(pipe, component)
            candidate = _artifact_block_weights(
                artifact_path,
                component,
                0,
                str(next(transformer.parameters()).device),
            )
            for name, module in transformer.blocks[0].named_modules():
                full_name = f"blocks.0.{name}"
                if full_name not in candidate:
                    continue
                packed, scales = quantize_mxfp4(module.weight.detach())
                rtn_weight = dequantize_mxfp4(packed, scales).to(
                    dtype=torch.bfloat16
                )
                self.handles.append(
                    module.register_forward_pre_hook(
                        self._hook(
                            component,
                            full_name,
                            module.weight.detach(),
                            rtn_weight,
                            candidate[full_name],
                            module.bias,
                        )
                    )
                )

    def _hook(self, component, name, reference_weight, rtn_weight, ptq_weight, bias):
        def hook(_module, inputs):
            values = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
            if values.shape[0] > self.rows:
                indices = torch.linspace(
                    0, values.shape[0] - 1, self.rows, device=values.device
                ).round().long()
                values = values.index_select(0, indices)
            reference = F.linear(values, reference_weight, bias)
            act_packed, act_scales = quantize_mxfp4(values)
            quantized_values = dequantize_mxfp4(act_packed, act_scales).to(
                dtype=values.dtype
            )
            call = self.calls[(component, name)]
            self.calls[(component, name)] += 1
            bucket = f"calls_{(call // 4) * 4:02d}_{(call // 4) * 4 + 3:02d}"
            for arm, weight in (("rtn", rtn_weight), ("ptq", ptq_weight)):
                output = F.linear(values, weight, bias)
                self.sums[(component, name, f"{bucket}:{arm}:bf16_act")].update(
                    reference, output
                )
                output = F.linear(quantized_values, weight, bias)
                self.sums[(component, name, f"{bucket}:{arm}:mxfp4_act")].update(
                    reference, output
                )

        return hook

    def remove(self) -> list[dict]:
        for handle in self.handles:
            handle.remove()
        reports = []
        for (expert, layer, key), sums in sorted(self.sums.items()):
            bucket, arm, activation = key.split(":")
            reports.append(
                {
                    "expert": expert,
                    "layer": layer,
                    "timestep_bucket": bucket,
                    "arm": arm,
                    "activation": activation,
                    **sums.report(),
                }
            )
        return reports


class TraceHooks:
    def __init__(self, pipe):
        self.predictions = {"transformer": [], "transformer_2": []}
        self.blocks = {"transformer": defaultdict(list), "transformer_2": defaultdict(list)}
        self.handles = []
        for component in ("transformer", "transformer_2"):
            transformer = getattr(pipe, component)
            self.handles.append(
                transformer.register_forward_hook(self._prediction_hook(component))
            )
            for index, block in enumerate(transformer.blocks):
                self.handles.append(
                    block.register_forward_hook(self._block_hook(component, index))
                )

    def _prediction_hook(self, component):
        def hook(_module, _inputs, output):
            self.predictions[component].append(_tensor_output(output).detach().cpu())

        return hook

    def _block_hook(self, component, index):
        def hook(_module, _inputs, output):
            self.blocks[component][index].append(_sample_rows(_tensor_output(output)))

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


class AttentionDecompositionHooks:
    """Decompose attention error at fixed layers/calls over the full K/V context."""

    _BLOCKS = (0, 20, 39)
    _CALLS = {
        "transformer": frozenset((0, 6, 12)),
        "transformer_2": frozenset((0, 12, 24)),
    }

    def __init__(self, pipe, query_rows: int = 4):
        self.query_rows = query_rows
        self.calls = defaultdict(int)
        self.rows = []
        self.processors = []
        for component in ("transformer", "transformer_2"):
            transformer = getattr(pipe, component)
            for block_index in self._BLOCKS:
                processor = transformer.blocks[block_index].attn1.processor
                if processor.qkv_observer is not None:
                    raise RuntimeError("attention processor already has a Q/K/V observer")
                processor.qkv_observer = self._observer(component, block_index)
                self.processors.append(processor)

    def _observer(self, component, block_index):
        def observe(_attention, query, key, value):
            call = self.calls[(component, block_index)]
            self.calls[(component, block_index)] += 1
            if call not in self._CALLS[component]:
                return
            indices = torch.linspace(
                0,
                query.shape[1] - 1,
                min(self.query_rows, query.shape[1]),
                device=query.device,
            ).round().long()
            outputs = mxfp4_attention_decomposition(
                query,
                key,
                value,
                query_indices=indices,
            )
            reference = outputs.pop("reference")
            errors = {}
            for arm, output in outputs.items():
                sums = ErrorSums()
                sums.update(reference, output)
                errors[arm] = sums.report()

            def stats(tensor):
                values = tensor.float()
                return {
                    "rms": values.square().mean().sqrt().item(),
                    "absmax": values.abs().amax().item(),
                }

            self.rows.append(
                {
                    "expert": component,
                    "block": block_index,
                    "call": call,
                    "query_indices": indices.detach().cpu().tolist(),
                    "shape": {
                        "q": list(query.shape),
                        "k": list(key.shape),
                        "v": list(value.shape),
                    },
                    "input_stats": {
                        "q": stats(query),
                        "k": stats(key),
                        "v": stats(value),
                    },
                    "errors": errors,
                }
            )

        return observe

    def remove(self) -> list[dict]:
        for processor in self.processors:
            processor.qkv_observer = None
        return self.rows


def _eval_args(
    arm: str,
    artifact_path: Path | None,
    *,
    gemm_mode: str | None = None,
    attention_artifact_path: Path | None = None,
):
    legacy_gemm_mode = {
        "bf16": "bf16",
        "rtn": "rtn",
        "ptq": "ptq",
        "attn_bf16": "bf16",
        "attn_mxfp4": "bf16",
    }[arm]
    if gemm_mode is None:
        gemm_mode = legacy_gemm_mode
    elif arm in ("bf16", "rtn", "ptq") and gemm_mode != legacy_gemm_mode:
        raise ValueError(
            f"legacy {arm} arm requires gemm mode {legacy_gemm_mode}; "
            "use an attention arm to vary GEMMs"
        )
    if gemm_mode not in ("bf16", "rtn", "ptq"):
        raise ValueError(f"unsupported GEMM mode: {gemm_mode}")
    attention_backend = {
        "attn_bf16": "AITER_BF16",
        "attn_mxfp4": "AITER_MXFP4",
    }.get(arm, "AITER_MXFP4")
    settings = CalibrationSettings(
        prompts_path=Path("unused"),
        attention_backend=attention_backend,
    )
    args = build_eager_bf16_args(
        settings,
        1,
        gemm_quantization="fp4" if gemm_mode in ("rtn", "ptq") else None,
        ptq_artifact_path=(
            str(artifact_path) if gemm_mode == "ptq" and artifact_path else None
        ),
        ptq_attention_artifact_path=(
            str(attention_artifact_path) if attention_artifact_path else None
        ),
    )
    args.output_type = "latent"
    if gemm_mode == "ptq":
        if artifact_path is None:
            raise ValueError("the PTQ GEMM mode requires --artifact")
    return args


class StepTimer:
    """Record synchronized callback-to-callback intervals, excluding warmup."""

    def __init__(self):
        self._last: float | None = None
        self.intervals_s: list[float] = []

    def mark(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self._last is not None:
            self.intervals_s.append(now - self._last)
        self._last = now

    def report(self) -> dict:
        return {
            "definition": "synchronized callback intervals after step 0",
            "values_s": self.intervals_s,
            "mean_s": (
                statistics.fmean(self.intervals_s) if self.intervals_s else None
            ),
            "median_s": (
                statistics.median(self.intervals_s) if self.intervals_s else None
            ),
        }


def _callback(
    latents: list[torch.Tensor],
    forced: list[torch.Tensor] | None,
    timer: StepTimer | None = None,
):
    def callback(_pipe, step, _timestep, callback_kwargs):
        if timer is not None:
            timer.mark()
        current = callback_kwargs["latents"]
        latents.append(current.detach().cpu())
        if forced is not None:
            callback_kwargs["latents"] = forced[step].to(
                device=current.device, dtype=current.dtype
            )
        return callback_kwargs

    return callback


def run_arm(
    *,
    arm: str,
    prompts_path: Path,
    calibration_prompts_path: Path,
    fixed_latent_path: Path,
    output_root: Path,
    artifact_path: Path | None,
    attention_artifact_path: Path | None = None,
    variant_label: str | None = None,
    gemm_mode: str | None = None,
    reference_dir: Path | None,
    max_prompts: int,
    prompt_offset: int,
    force_reference_trajectory: bool,
    layer_metrics: bool,
    attention_decomposition: bool,
) -> Path:
    from xfuser.core.distributed import init_distributed_environment
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    if not torch.distributed.is_initialized():
        init_distributed_environment()
    if torch.distributed.get_world_size() != 1:
        raise RuntimeError("Phase-0 accuracy traces require one data-parallel rank")

    calibration_prompts = read_prompts(calibration_prompts_path)
    calibration_set = set(calibration_prompts)
    prompts = [
        prompt for prompt in read_prompts(prompts_path)
        if prompt not in calibration_set
    ][prompt_offset : prompt_offset + max_prompts]
    if len(prompts) < max_prompts:
        raise ValueError("not enough held-out prompts after excluding calibration prompts")
    assert_no_prompt_overlap(calibration_prompts, prompts)
    if attention_decomposition and (
        arm not in ("attn_bf16", "attn_mxfp4") or len(prompts) != 1
    ):
        raise ValueError(
            "attention decomposition requires one attention-arm prompt"
        )
    args = _eval_args(
        arm,
        artifact_path,
        gemm_mode=gemm_mode,
        attention_artifact_path=attention_artifact_path,
    )
    resolved_gemm_mode = (
        "ptq"
        if args.ptq_artifact_path
        else "rtn"
        if args.gemm_quantization_spec.is_pure("fp4")
        else "bf16"
    )
    if variant_label is None:
        variant_label = (
            "plain"
            if attention_artifact_path is None
            else _validated_attention_artifact(
                str(attention_artifact_path)
            ).artifact_id
        )
    model = MODEL_REGISTRY[MODEL_ID](args)
    input_args = {
        "height": 720,
        "width": 1280,
        "num_frames": 81,
        "num_inference_steps": 20,
        "guidance_scale": 4.0,
        "guidance_scale_2": 3.0,
        "flow_shift": 3.0,
        "prompt": prompts[0],
        "negative_prompt": model.default_input_values.negative_prompt,
        "seed": 42,
    }
    model.initialize(input_args)
    fixed_latent = torch.load(
        fixed_latent_path, map_location="cpu", weights_only=True
    )

    run_payload = {
        "arm": arm,
        "artifact_id": None,
        "gemm_mode": resolved_gemm_mode,
        "variant_label": variant_label,
        "attention_artifact_id": None,
        "attention_artifact_path": (
            str(attention_artifact_path) if attention_artifact_path else None
        ),
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompts_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "prompt_offset": prompt_offset,
        "prompt_count": len(prompts),
        "fixed_latent_sha256": hashlib.sha256(fixed_latent_path.read_bytes()).hexdigest(),
        "force_reference_trajectory": force_reference_trajectory,
        "attention_backend": args.attention_backend,
        "prompts": [],
    }
    if artifact_path:
        artifact = _validated_artifact(str(artifact_path))
        run_payload["artifact_id"] = artifact.artifact_id
    if attention_artifact_path:
        attention_artifact = _validated_attention_artifact(
            str(attention_artifact_path)
        )
        run_payload["attention_artifact_id"] = attention_artifact.artifact_id
    digest = hashlib.sha256(
        json.dumps(run_payload, sort_keys=True).encode()
    ).hexdigest()[:12]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{arm}-{digest}"
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    layer_hook = (
        LayerErrorHooks(model.pipe, artifact_path)
        if layer_metrics and arm == "bf16" and artifact_path
        else None
    )
    attention_hook = (
        AttentionDecompositionHooks(model.pipe)
        if attention_decomposition
        else None
    )
    try:
        for local_index, prompt in enumerate(prompts):
            prompt_index = prompt_offset + local_index
            forced = None
            if force_reference_trajectory:
                if reference_dir is None:
                    raise ValueError("--force-reference-trajectory requires --reference-dir")
                reference = torch.load(
                    reference_dir / f"prompt-{prompt_index:03d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                forced = reference["latents"]
            trace = TraceHooks(model.pipe)
            latents = []
            timer = StepTimer()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            generator = torch.Generator(
                device=model.pipe._execution_device
            ).manual_seed(42 + prompt_index)
            output = model.pipe(
                height=720,
                width=1280,
                num_frames=81,
                num_inference_steps=20,
                prompt=prompt,
                negative_prompt=model.default_input_values.negative_prompt,
                guidance_scale=4.0,
                guidance_scale_2=3.0,
                generator=generator,
                latents=fixed_latent.to(model.pipe._execution_device),
                output_type=(
                    "latent" if force_reference_trajectory else "np"
                ),
                callback_on_step_end=_callback(latents, forced, timer),
                callback_on_step_end_tensor_inputs=["latents"],
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_memory_bytes = torch.cuda.max_memory_allocated()
            else:
                peak_memory_bytes = None
            trace.remove()
            if not force_reference_trajectory:
                frames = output.frames if hasattr(output, "frames") else output[0]
                np.savez_compressed(
                    output_dir / f"prompt-{prompt_index:03d}-video.npz",
                    frames=np.asarray(frames),
                )
            torch.save(
                {
                    "latents": latents,
                    "predictions": trace.predictions,
                    "blocks": {
                        expert: dict(values)
                        for expert, values in trace.blocks.items()
                    },
                },
                output_dir / f"prompt-{prompt_index:03d}.pt",
            )
            run_payload["prompts"].append(
                {
                    "index": prompt_index,
                    "text": prompt,
                    "seed": 42 + prompt_index,
                    "performance": {
                        "steady_step_time": timer.report(),
                        "peak_device_memory_bytes": peak_memory_bytes,
                    },
                }
            )
    finally:
        if layer_hook is not None:
            reports = layer_hook.remove()
            (output_dir / "rung1-layer-error.json").write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if attention_hook is not None:
            reports = attention_hook.remove()
            (output_dir / "attention-decomposition.json").write_text(
                json.dumps(reports, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {**run_payload, "runtime": runtime_manifest()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_dir


def _compare_lists(reference, candidate) -> dict:
    sums = ErrorSums()
    if len(reference) != len(candidate):
        raise ValueError(f"trace length mismatch: {len(reference)} != {len(candidate)}")
    for expected, actual in zip(reference, candidate):
        sums.update(expected, actual)
    return sums.report()


def _velocity_metrics(reference, candidate, guidance: float) -> dict:
    if len(reference) != len(candidate):
        raise ValueError("velocity trace length mismatch")
    if not reference:
        return {"branches": [], "cfg": [], "overall": ErrorSums().report()}
    steps = []
    if reference[0].shape[0] >= 2:
        steps = [
            ((expected[0], expected[1]), (actual[0], actual[1]))
            for expected, actual in zip(reference, candidate)
        ]
    else:
        if len(reference) % 2:
            raise ValueError("separate-CFG velocity trace has an odd call count")
        for index in range(0, len(reference), 2):
            # WanPipeline calls the conditional branch first, then the
            # unconditional branch when CFG is not parallelized.
            steps.append(
                (
                    (reference[index + 1], reference[index]),
                    (candidate[index + 1], candidate[index]),
                )
            )
    selected = sorted(
        set(
            torch.linspace(0, len(steps) - 1, min(5, len(steps)))
            .round()
            .int()
            .tolist()
        )
    )
    reports = []
    overall = ErrorSums()
    for step in selected:
        (ref_uncond, ref_cond), (cand_uncond, cand_cond) = steps[step]
        uncond = ErrorSums()
        cond = ErrorSums()
        cfg = ErrorSums()
        uncond.update(ref_uncond, cand_uncond)
        cond.update(ref_cond, cand_cond)
        ref_cfg = ref_uncond + guidance * (ref_cond - ref_uncond)
        cand_cfg = cand_uncond + guidance * (cand_cond - cand_uncond)
        cfg.update(ref_cfg, cand_cfg)
        overall.update(ref_cfg, cand_cfg)
        reports.append(
            {
                "step_within_expert": step,
                "unconditional": uncond.report(),
                "conditional": cond.report(),
                "cfg_combined": cfg.report(),
            }
        )
    return {"sampled_steps": reports, "cfg_combined": overall.report()}


def _ssim_frames(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    # Standard local SSIM with an 11x11 uniform window; inputs are NCHW in [0, 1].
    c1, c2 = 0.01**2, 0.03**2
    mu_x = F.avg_pool2d(reference, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(candidate, 11, stride=1, padding=5)
    var_x = F.avg_pool2d(reference.square(), 11, 1, 5) - mu_x.square()
    var_y = F.avg_pool2d(candidate.square(), 11, 1, 5) - mu_y.square()
    cov = F.avg_pool2d(reference * candidate, 11, 1, 5) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return value.mean().item()


def _decomposition_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    by_arm = defaultdict(list)
    for row in rows:
        for arm, metrics in row["errors"].items():
            by_arm[arm].append(metrics["relative_mse"])
    return {
        "path": str(path),
        "measurements": len(rows),
        "relative_mse": {
            arm: _distribution(values) for arm, values in sorted(by_arm.items())
        },
    }


def compare_runs(reference_dir: Path, arm_dirs: list[Path], output: Path) -> dict:
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    result = {"reference": str(reference_dir), "arms": {}}
    layer_report_path = reference_dir / "rung1-layer-error.json"
    if layer_report_path.exists():
        layer_rows = json.loads(layer_report_path.read_text(encoding="utf-8"))
        by_key = defaultdict(dict)
        for row in layer_rows:
            key = (
                row["expert"],
                row["layer"],
                row["timestep_bucket"],
                row["activation"],
            )
            by_key[key][row["arm"]] = row
        result["rung1"] = []
        for key, arms in sorted(by_key.items()):
            if "rtn" not in arms or "ptq" not in arms:
                continue
            rtn_error = arms["rtn"]["relative_mse"]
            ptq_error = arms["ptq"]["relative_mse"]
            result["rung1"].append(
                {
                    "expert": key[0],
                    "layer": key[1],
                    "timestep_bucket": key[2],
                    "activation": key[3],
                    "rtn": arms["rtn"],
                    "ptq": arms["ptq"],
                    "gap_closed": (rtn_error - ptq_error) / max(rtn_error, 1.0e-30),
                }
            )
    grouped = defaultdict(dict)
    for arm_dir in arm_dirs:
        manifest = json.loads((arm_dir / "manifest.json").read_text(encoding="utf-8"))
        mode = "forced" if manifest["force_reference_trajectory"] else "free"
        label = manifest["arm"]
        if manifest.get("attention_artifact_id"):
            label = manifest.get("variant_label") or manifest["attention_artifact_id"]
        gemm_mode = manifest.get(
            "gemm_mode",
            {"rtn": "rtn", "ptq": "ptq"}.get(manifest["arm"], "bf16"),
        )
        key = (label, gemm_mode, manifest["arm"])
        if mode in grouped[key]:
            raise ValueError(f"duplicate {mode} run for {key}")
        grouped[key][mode] = (arm_dir, manifest)
    for (label, gemm_mode, arm), modes in grouped.items():
        if "forced" not in modes or "free" not in modes:
            raise ValueError(
                f"{label}/{gemm_mode}/{arm} requires both forced and free "
                "trajectory runs"
            )
        forced_dir, forced_manifest = modes["forced"]
        free_dir, free_manifest = modes["free"]
        if (
            forced_manifest.get("attention_artifact_id")
            != free_manifest.get("attention_artifact_id")
            or forced_manifest.get("variant_label")
            != free_manifest.get("variant_label")
        ):
            raise ValueError(
                f"{label}/{gemm_mode}/{arm} forced/free provenance mismatch"
            )
        prompts = []
        for prompt in reference_manifest["prompts"]:
            index = prompt["index"]
            reference = torch.load(
                reference_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            candidate = torch.load(
                forced_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            free_candidate = torch.load(
                free_dir / f"prompt-{index:03d}.pt",
                map_location="cpu",
                weights_only=True,
            )
            block_metrics = {}
            velocity_metrics = {}
            for expert in ("transformer", "transformer_2"):
                velocity_metrics[expert] = _velocity_metrics(
                    reference["predictions"][expert],
                    candidate["predictions"][expert],
                    4.0 if expert == "transformer" else 3.0,
                )
                for block_index, expected in reference["blocks"][expert].items():
                    block_metrics[f"{expert}.blocks.{block_index}"] = _compare_lists(
                        expected,
                        candidate["blocks"][expert][block_index],
                    )
            latent_metrics = [
                _compare_lists([expected], [actual])
                for expected, actual in zip(reference["latents"], free_candidate["latents"])
            ]
            reference_video = np.load(
                reference_dir / f"prompt-{index:03d}-video.npz"
            )["frames"]
            candidate_video = np.load(
                free_dir / f"prompt-{index:03d}-video.npz"
            )["frames"]
            ref_tensor = torch.from_numpy(reference_video).float()
            cand_tensor = torch.from_numpy(candidate_video).float()
            if ref_tensor.max() > 1:
                ref_tensor /= 255
                cand_tensor /= 255
            images_ref = ref_tensor.reshape(-1, *ref_tensor.shape[-3:]).permute(0, 3, 1, 2)
            images_cand = cand_tensor.reshape(-1, *cand_tensor.shape[-3:]).permute(0, 3, 1, 2)
            mse = (images_ref - images_cand).square().flatten(1).mean(1)
            prompts.append(
                {
                    "index": index,
                    "rung2": {
                        "block0_bf16_input": {
                            key: value
                            for key, value in block_metrics.items()
                            if key.endswith(".blocks.0")
                        },
                        "propagated_blocks": block_metrics,
                    },
                    "rung3": {"velocity": velocity_metrics},
                    "rung4": {
                        "latent_drift": latent_metrics,
                        "frame_psnr_db": (
                            -10 * torch.log10(mse.clamp_min(1.0e-30))
                        ).tolist(),
                        "frame_ssim": _ssim_frames(images_ref, images_cand),
                        "lpips": None,
                        "clip_text_video_similarity": None,
                    },
                }
            )
        output_label = label
        if output_label in result["arms"]:
            output_label = f"{label}@{gemm_mode}"
        result["arms"][output_label] = {
            "forced_run": str(forced_dir),
            "free_run": str(free_dir),
            "artifact_id": free_manifest.get("artifact_id"),
            "attention_artifact_id": free_manifest.get("attention_artifact_id"),
            "variant_label": free_manifest.get("variant_label", label),
            "gemm_mode": gemm_mode,
            "attention_backend": free_manifest.get("attention_backend"),
            "arm": free_manifest["arm"],
            "performance": [
                prompt.get("performance") for prompt in free_manifest["prompts"]
            ],
            "attention_emulator_decomposition": (
                _decomposition_metrics(
                    forced_dir / "attention-decomposition.json"
                )
                or _decomposition_metrics(
                    free_dir / "attention-decomposition.json"
                )
            ),
            "prompts": prompts,
        }
    if "rtn" in result["arms"] and "ptq" in result["arms"]:
        gaps = []
        for rtn_prompt, ptq_prompt in zip(
            result["arms"]["rtn"]["prompts"],
            result["arms"]["ptq"]["prompts"],
        ):
            entry = {
                "index": rtn_prompt["index"],
                "rung2": {},
                "rung3": {},
                "rung4": [],
            }
            for block, rtn_metric in rtn_prompt["rung2"]["propagated_blocks"].items():
                ptq_metric = ptq_prompt["rung2"]["propagated_blocks"][block]
                entry["rung2"][block] = (
                    rtn_metric["relative_mse"] - ptq_metric["relative_mse"]
                ) / max(rtn_metric["relative_mse"], 1.0e-30)
            for expert, rtn_metric in rtn_prompt["rung3"]["velocity"].items():
                ptq_metric = ptq_prompt["rung3"]["velocity"][expert]
                entry["rung3"][expert] = (
                    rtn_metric["cfg_combined"]["relative_mse"]
                    - ptq_metric["cfg_combined"]["relative_mse"]
                ) / max(rtn_metric["cfg_combined"]["relative_mse"], 1.0e-30)
            for rtn_metric, ptq_metric in zip(
                rtn_prompt["rung4"]["latent_drift"],
                ptq_prompt["rung4"]["latent_drift"],
            ):
                entry["rung4"].append(
                    (rtn_metric["relative_mse"] - ptq_metric["relative_mse"])
                    / max(rtn_metric["relative_mse"], 1.0e-30)
                )
            gaps.append(entry)
        result["gap_closed"] = gaps

    # Stage 3 compares every transformed MXFP4-attention candidate with the
    # untransformed MXFP4-attention run using the identical GEMM mode.
    plain_by_gemm = {}
    for label, arm_result in result["arms"].items():
        if (
            arm_result["arm"] == "attn_mxfp4"
            and arm_result["attention_artifact_id"] is None
        ):
            key = (arm_result["gemm_mode"], arm_result["artifact_id"])
            if key in plain_by_gemm:
                raise ValueError(f"duplicate plain MXFP4 comparator for {key}")
            plain_by_gemm[key] = arm_result
    candidate_gaps = []
    for label, arm_result in result["arms"].items():
        if (
            arm_result["arm"] != "attn_mxfp4"
            or arm_result["attention_artifact_id"] is None
        ):
            continue
        comparator_key = (arm_result["gemm_mode"], arm_result["artifact_id"])
        comparator = plain_by_gemm.get(comparator_key)
        if comparator is None:
            raise ValueError(
                f"{label} has no plain MXFP4 comparator with GEMM mode "
                f"{arm_result['gemm_mode']} and GEMM artifact "
                f"{arm_result['artifact_id']}"
            )
        if len(comparator["prompts"]) != len(arm_result["prompts"]):
            raise ValueError("candidate/comparator prompt-count mismatch")
        prompt_gaps = []
        for plain_prompt, candidate_prompt in zip(
            comparator["prompts"], arm_result["prompts"]
        ):
            if plain_prompt["index"] != candidate_prompt["index"]:
                raise ValueError("candidate/comparator prompt mismatch")
            rung3 = {}
            for expert in ("transformer", "transformer_2"):
                baseline = plain_prompt["rung3"]["velocity"][expert][
                    "cfg_combined"
                ]["relative_mse"]
                candidate = candidate_prompt["rung3"]["velocity"][expert][
                    "cfg_combined"
                ]["relative_mse"]
                rung3[expert] = (baseline - candidate) / max(baseline, 1.0e-30)
            rung3["equal_mean"] = statistics.fmean(rung3.values())
            baseline_final = plain_prompt["rung4"]["latent_drift"][-1][
                "relative_mse"
            ]
            candidate_final = candidate_prompt["rung4"]["latent_drift"][-1][
                "relative_mse"
            ]
            prompt_gaps.append(
                {
                    "index": plain_prompt["index"],
                    "rung3": rung3,
                    "rung4": {
                        "final_latent": (
                            (baseline_final - candidate_final)
                            / max(baseline_final, 1.0e-30)
                        )
                    },
                }
            )
        candidate_gaps.append(
            {
                "variant_label": arm_result["variant_label"],
                "attention_artifact_id": arm_result["attention_artifact_id"],
                "gemm_mode": arm_result["gemm_mode"],
                "comparator": "plain_mxfp4",
                "prompts": prompt_gaps,
            }
        )
    if candidate_gaps:
        result["candidate_gap_closed"] = candidate_gaps

    t0 = []
    for label, arm_result in result["arms"].items():
        if (
            arm_result["arm"] != "attn_bf16"
            or arm_result["attention_artifact_id"] is None
        ):
            continue
        prompts = []
        for prompt in arm_result["prompts"]:
            block_metrics = prompt["rung2"]["propagated_blocks"]
            velocity_metrics = prompt["rung3"]["velocity"]
            measured = [
                *block_metrics.values(),
                *(
                    velocity_metrics[expert]["cfg_combined"]
                    for expert in ("transformer", "transformer_2")
                ),
            ]
            max_relative_mse = max(row["relative_mse"] for row in measured)
            min_cosine = min(row["cosine_similarity"] for row in measured)
            prompts.append(
                {
                    "index": prompt["index"],
                    "max_relative_mse": max_relative_mse,
                    "min_cosine_similarity": min_cosine,
                    "passes": max_relative_mse <= 1.0e-4 and min_cosine >= 0.999,
                }
            )
        t0.append(
            {
                "variant_label": arm_result["variant_label"],
                "attention_artifact_id": arm_result["attention_artifact_id"],
                "gemm_mode": arm_result["gemm_mode"],
                "tolerance": {
                    "max_relative_mse": 1.0e-4,
                    "min_cosine_similarity": 0.999,
                },
                "prompts": prompts,
                "passes": all(prompt["passes"] for prompt in prompts),
            }
        )
    if t0:
        result["function_preservation"] = t0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "values": values,
    }


def _aggregate_stage3_reports(
    reports: list[dict], report_paths: list[Path], output: Path
) -> dict:
    grouped: dict[tuple[str, str], dict] = {}
    function_preservation = defaultdict(list)
    for report in reports:
        for candidate in report.get("function_preservation", []):
            key = (candidate["gemm_mode"], candidate["variant_label"])
            function_preservation[key].extend(candidate["prompts"])
        for candidate in report.get("candidate_gap_closed", []):
            key = (candidate["gemm_mode"], candidate["variant_label"])
            entry = grouped.setdefault(
                key,
                {
                    "artifact_ids": set(),
                    "rung3": defaultdict(list),
                    "rung4": [],
                    "step_times": [],
                    "peak_memory": [],
                    "decomposition": defaultdict(list),
                },
            )
            entry["artifact_ids"].add(candidate["attention_artifact_id"])
            for prompt in candidate["prompts"]:
                for expert in ("transformer", "transformer_2", "equal_mean"):
                    entry["rung3"][expert].append(prompt["rung3"][expert])
                entry["rung4"].append(prompt["rung4"]["final_latent"])
            for arm in report["arms"].values():
                if (
                    arm.get("gemm_mode") == candidate["gemm_mode"]
                    and arm.get("variant_label") == candidate["variant_label"]
                ):
                    decomposition = arm.get("attention_emulator_decomposition")
                    if decomposition:
                        for name, metrics in decomposition[
                            "relative_mse"
                        ].items():
                            entry["decomposition"][name].extend(
                                metrics["values"]
                            )
                    for performance in arm.get("performance", []):
                        if not performance:
                            continue
                        mean = performance["steady_step_time"].get("mean_s")
                        if mean is not None:
                            entry["step_times"].append(mean)
                        peak = performance.get("peak_device_memory_bytes")
                        if peak is not None:
                            entry["peak_memory"].append(peak)

    variants = {}
    for (gemm_mode, variant_label), entry in sorted(grouped.items()):
        variants.setdefault(gemm_mode, {})[variant_label] = {
            "attention_artifact_ids": sorted(entry["artifact_ids"]),
            "prompt_count": len(entry["rung4"]),
            "gap_closed": {
                "rung3": {
                    expert: _distribution(entry["rung3"][expert])
                    for expert in ("transformer", "transformer_2", "equal_mean")
                },
                "rung4": {"final_latent": _distribution(entry["rung4"])},
            },
            "performance": {
                "steady_step_time_s": (
                    _distribution(entry["step_times"])
                    if entry["step_times"]
                    else None
                ),
                "peak_device_memory_bytes": (
                    _distribution(entry["peak_memory"])
                    if entry["peak_memory"]
                    else None
                ),
            },
            "attention_emulator_decomposition": {
                name: _distribution(values)
                for name, values in sorted(entry["decomposition"].items())
            },
        }

    selections = {}
    for gemm_mode, mode_variants in variants.items():
        eligible = [
            label for label in REGISTERED_VARIANTS if label in mode_variants
        ]
        if len(eligible) != len(REGISTERED_VARIANTS):
            selections[gemm_mode] = {
                "status": "incomplete",
                "rule": (
                    "run all four registered variants before selection"
                ),
                "eligible_variants": eligible,
                "missing_variants": [
                    label for label in REGISTERED_VARIANTS
                    if label not in mode_variants
                ],
            }
            continue
        selected = max(
            eligible,
            key=lambda label: (
                mode_variants[label]["gap_closed"]["rung3"]["equal_mean"]["mean"],
                mode_variants[label]["gap_closed"]["rung4"]["final_latent"]["mean"],
            ),
        )
        power_of_two_variant = (
            f"{selected}-pow2"
            if "t3" in selected
            else f"{selected}+t3-pow2"
        )
        selections[gemm_mode] = {
            "status": "selected",
            "rule": (
                "maximize equal-weight mean rung-3 gap closed; "
                "break ties by mean final-latent rung-4 gap closed"
            ),
            "eligible_variants": eligible,
            "selected_variant": selected,
            "power_of_two_followup_variant": power_of_two_variant,
        }

    function_summary = {}
    for (gemm_mode, variant_label), prompts in sorted(
        function_preservation.items()
    ):
        function_summary.setdefault(gemm_mode, {})[variant_label] = {
            "prompt_count": len(prompts),
            "max_relative_mse": max(
                prompt["max_relative_mse"] for prompt in prompts
            ),
            "min_cosine_similarity": min(
                prompt["min_cosine_similarity"] for prompt in prompts
            ),
            "passes": all(prompt["passes"] for prompt in prompts),
        }

    summary = {
        "reports": [str(path) for path in report_paths],
        "registered_variants": list(REGISTERED_VARIANTS),
        "variants_by_gemm_mode": variants,
        "selection_by_gemm_mode": selections,
        "function_preservation_by_gemm_mode": function_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_reports(report_paths: list[Path], output: Path) -> dict:
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    if any(
        "candidate_gap_closed" in report or "function_preservation" in report
        for report in reports
    ):
        return _aggregate_stage3_reports(reports, report_paths, output)
    gaps = [report["gap_closed"][0] for report in reports]

    def mean_at(path):
        values = []
        for item in gaps:
            value = item
            for key in path:
                value = value[key]
            values.append(value)
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "values": values,
        }

    summary = {
        "reports": [str(path) for path in report_paths],
        "prompt_count": len(reports),
        "gap_closed": {
            "rung2": {
                name: mean_at(("rung2", name))
                for name in (
                    "transformer.blocks.0",
                    "transformer.blocks.39",
                    "transformer_2.blocks.0",
                    "transformer_2.blocks.39",
                )
            },
            "rung3": {
                expert: mean_at(("rung3", expert))
                for expert in ("transformer", "transformer_2")
            },
            "rung4_by_step": [
                {
                    "mean": statistics.fmean(
                        item["rung4"][step] for item in gaps
                    ),
                    "median": statistics.median(
                        item["rung4"][step] for item in gaps
                    ),
                }
                for step in range(len(gaps[0]["rung4"]))
            ],
        },
        "arms": {},
    }
    for arm in ("rtn", "ptq"):
        prompts = [report["arms"][arm]["prompts"][0] for report in reports]
        ssim = [prompt["rung4"]["frame_ssim"] for prompt in prompts]
        psnr = [
            statistics.fmean(prompt["rung4"]["frame_psnr_db"])
            for prompt in prompts
        ]
        final_latent = [
            prompt["rung4"]["latent_drift"][-1]["relative_mse"]
            for prompt in prompts
        ]
        summary["arms"][arm] = {
            "frame_ssim": {
                "mean": statistics.fmean(ssim),
                "median": statistics.median(ssim),
                "values": ssim,
            },
            "frame_psnr_db": {
                "mean": statistics.fmean(psnr),
                "median": statistics.median(psnr),
                "values": psnr,
            },
            "final_latent_relative_mse": {
                "mean": statistics.fmean(final_latent),
                "median": statistics.median(final_latent),
                "values": final_latent,
            },
        }
    rung1 = reports[0].get("rung1")
    if rung1:
        summary["rung1"] = {}
        for activation in ("bf16_act", "mxfp4_act"):
            values = [
                row["gap_closed"]
                for row in rung1
                if row["activation"] == activation
            ]
            summary["rung1"][activation] = {
                "mean_gap_closed": statistics.fmean(values),
                "median_gap_closed": statistics.median(values),
                "measurements": len(values),
            }
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def aggregate_attention_reports(
    report_paths: list[Path],
    rtn_report_paths: list[Path],
    decomposition_path: Path,
    output: Path,
) -> dict:
    """Aggregate Stage 0B attention errors and apply its pre-registered gate."""

    if len(report_paths) != len(rtn_report_paths):
        raise ValueError("attention and RTN report counts must match")
    reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in report_paths
    ]
    rtn_reports = [
        json.loads(path.read_text(encoding="utf-8")) for path in rtn_report_paths
    ]
    candidate_rung3 = defaultdict(list)
    rtn_rung3 = defaultdict(list)
    rung4 = []
    frame_ssim = []
    frame_psnr = []
    for report, rtn_report in zip(reports, rtn_reports):
        prompt = report["arms"]["attn_mxfp4"]["prompts"][0]
        rtn_prompt = rtn_report["arms"]["rtn"]["prompts"][0]
        for expert in ("transformer", "transformer_2"):
            candidate_rung3[expert].append(
                prompt["rung3"]["velocity"][expert]["cfg_combined"]["relative_mse"]
            )
            rtn_rung3[expert].append(
                rtn_prompt["rung3"]["velocity"][expert]["cfg_combined"]["relative_mse"]
            )
        rung4.append(prompt["rung4"]["latent_drift"][-1]["relative_mse"])
        frame_ssim.append(prompt["rung4"]["frame_ssim"])
        frame_psnr.append(statistics.fmean(prompt["rung4"]["frame_psnr_db"]))

    candidate_values = [
        value for expert in candidate_rung3.values() for value in expert
    ]
    rtn_values = [value for expert in rtn_rung3.values() for value in expert]
    rung3_ratio = statistics.fmean(candidate_values) / max(
        statistics.fmean(rtn_values), 1.0e-30
    )

    decomposition_rows = json.loads(
        decomposition_path.read_text(encoding="utf-8")
    )
    isolated = defaultdict(list)
    for row in decomposition_rows:
        for arm, metrics in row["errors"].items():
            isolated[arm].append(metrics["relative_mse"])
    isolated_means = {
        arm: statistics.fmean(values) for arm, values in sorted(isolated.items())
    }
    isolated_sum = sum(isolated_means[arm] for arm in ("q_only", "k_only", "v_only", "p_only"))
    p_fraction = isolated_means["p_only"] / max(isolated_sum, 1.0e-30)
    p_dominates = isolated_means["p_only"] >= max(
        isolated_means[arm] for arm in ("q_only", "k_only", "v_only")
    ) and p_fraction >= 0.5
    error_too_small = rung3_ratio < 0.1

    summary = {
        "reports": [str(path) for path in report_paths],
        "rtn_reports": [str(path) for path in rtn_report_paths],
        "prompt_count": len(reports),
        "rung3": {
            "attention_relative_mse": {
                expert: {
                    "mean": statistics.fmean(values),
                    "values": values,
                }
                for expert, values in candidate_rung3.items()
            },
            "rtn_gemm_relative_mse": {
                expert: {
                    "mean": statistics.fmean(values),
                    "values": values,
                }
                for expert, values in rtn_rung3.items()
            },
            "mean_error_ratio_to_rtn_gemm": rung3_ratio,
        },
        "rung4": {
            "final_latent_relative_mse": {
                "mean": statistics.fmean(rung4),
                "values": rung4,
            },
            "frame_ssim": {
                "mean": statistics.fmean(frame_ssim),
                "values": frame_ssim,
            },
            "frame_psnr_db": {
                "mean": statistics.fmean(frame_psnr),
                "values": frame_psnr,
            },
        },
        "decomposition": {
            "path": str(decomposition_path),
            "measurements": len(decomposition_rows),
            "mean_relative_mse": isolated_means,
            "p_fraction_of_isolated_error_sum": p_fraction,
            "p_dominates": p_dominates,
        },
        "gate": {
            "error_under_ten_percent_of_rtn": error_too_small,
            "p_quantization_dominates": p_dominates,
            "decision": "stop" if error_too_small or p_dominates else "advance",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument(
        "--arm",
        choices=("bf16", "rtn", "ptq", "attn_bf16", "attn_mxfp4"),
        required=True,
    )
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--calibration-prompts", type=Path, required=True)
    run.add_argument("--fixed-latent", type=Path, required=True)
    run.add_argument("--output-root", type=Path, default=Path("/data/ptq/eval"))
    run.add_argument("--artifact", type=Path)
    run.add_argument("--attention-artifact", type=Path)
    run.add_argument("--variant-label")
    run.add_argument("--gemm-mode", choices=("bf16", "rtn", "ptq"))
    run.add_argument("--reference-dir", type=Path)
    run.add_argument("--max-prompts", type=int, default=1)
    run.add_argument("--prompt-offset", type=int, default=0)
    run.add_argument("--force-reference-trajectory", action="store_true")
    run.add_argument("--layer-metrics", action="store_true")
    run.add_argument("--attention-decomposition", action="store_true")
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference-dir", type=Path, required=True)
    compare.add_argument("--arm-dir", type=Path, action="append", required=True)
    compare.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--report", type=Path, action="append", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate_attention = subparsers.add_parser("aggregate-attention")
    aggregate_attention.add_argument(
        "--report", type=Path, action="append", required=True
    )
    aggregate_attention.add_argument(
        "--rtn-report", type=Path, action="append", required=True
    )
    aggregate_attention.add_argument("--decomposition", type=Path, required=True)
    aggregate_attention.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    cli = _parse_args()
    if cli.command == "run":
        print(
            run_arm(
                arm=cli.arm,
                prompts_path=cli.prompts,
                calibration_prompts_path=cli.calibration_prompts,
                fixed_latent_path=cli.fixed_latent,
                output_root=cli.output_root,
                artifact_path=cli.artifact,
                attention_artifact_path=cli.attention_artifact,
                variant_label=cli.variant_label,
                gemm_mode=cli.gemm_mode,
                reference_dir=cli.reference_dir,
                max_prompts=cli.max_prompts,
                prompt_offset=cli.prompt_offset,
                force_reference_trajectory=cli.force_reference_trajectory,
                layer_metrics=cli.layer_metrics,
                attention_decomposition=cli.attention_decomposition,
            )
        )
    elif cli.command == "compare":
        print(compare_runs(cli.reference_dir, cli.arm_dir, cli.output)["arms"].keys())
    elif cli.command == "aggregate":
        print(aggregate_reports(cli.report, cli.output).get("selection_by_gemm_mode", {}))
    else:
        print(
            aggregate_attention_reports(
                cli.report,
                cli.rtn_report,
                cli.decomposition,
                cli.output,
            )["gate"]
        )
