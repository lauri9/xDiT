"""Compute LPIPS and CLIP metrics for completed PTQ ladder reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch


def _video(run_dir: Path, index: int) -> np.ndarray:
    frames = np.load(run_dir / f"prompt-{index:03d}-video.npz")["frames"]
    frames = frames.reshape(-1, *frames.shape[-3:])
    if frames.dtype == np.uint8:
        frames = frames.astype(np.float32) / 255.0
    return frames.astype(np.float32, copy=False)


def _report_inputs(report_path: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    reference_dir = Path(report["reference"])
    reference_manifest = json.loads(
        (reference_dir / "manifest.json").read_text(encoding="utf-8")
    )
    prompt = reference_manifest["prompts"][0]
    runs = {"bf16": reference_dir}
    runs.update(
        {
            arm: Path(details["free_run"])
            for arm, details in report["arms"].items()
        }
    )
    return prompt, runs


def compute_lpips(report_paths: list[Path], device: str) -> dict:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to(device)
    result = {"metric": "LPIPS/AlexNet", "prompts": []}
    for report_path in report_paths:
        prompt, runs = _report_inputs(report_path)
        reference = _video(runs["bf16"], prompt["index"])
        entry = {"index": prompt["index"], "arms": {}}
        for arm in ("rtn", "ptq"):
            candidate = _video(runs[arm], prompt["index"])
            scores = []
            for start in range(0, len(reference), 2):
                expected = torch.from_numpy(reference[start : start + 2]).permute(
                    0, 3, 1, 2
                ).to(device)
                actual = torch.from_numpy(candidate[start : start + 2]).permute(
                    0, 3, 1, 2
                ).to(device)
                metric.reset()
                scores.append(float(metric(expected, actual).item()))
            entry["arms"][arm] = {
                "mean": float(np.mean(scores)),
                "batch_scores": scores,
            }
        result["prompts"].append(entry)
    return result


def compute_clip(report_paths: list[Path], device: str, cache_dir: Path) -> dict:
    import clip
    from PIL import Image

    model, preprocess = clip.load(
        "ViT-B/32", device=device, download_root=str(cache_dir)
    )
    model.eval()
    result = {"metric": "CLIP ViT-B/32 text-video cosine", "prompts": []}
    for report_path in report_paths:
        prompt, runs = _report_inputs(report_path)
        text = clip.tokenize([prompt["text"]]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text).float()
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
        entry = {"index": prompt["index"], "arms": {}}
        for arm, run_dir in runs.items():
            frames = _video(run_dir, prompt["index"])
            features = []
            for start in range(0, len(frames), 16):
                images = torch.stack(
                    [
                        preprocess(
                            Image.fromarray(
                                (frame.clip(0, 1) * 255).round().astype(np.uint8)
                            )
                        )
                        for frame in frames[start : start + 16]
                    ]
                ).to(device)
                with torch.no_grad():
                    feature = model.encode_image(images).float()
                    feature /= feature.norm(dim=-1, keepdim=True)
                features.append(feature)
            video_feature = torch.cat(features).mean(dim=0, keepdim=True)
            video_feature /= video_feature.norm(dim=-1, keepdim=True)
            entry["arms"][arm] = float((video_feature @ text_feature.T).item())
        result["prompts"].append(entry)
    return result


def merge_metric_summaries(
    numerical_path: Path,
    lpips_path: Path,
    clip_path: Path,
    output_path: Path,
) -> dict:
    summary = json.loads(numerical_path.read_text(encoding="utf-8"))
    lpips = json.loads(lpips_path.read_text(encoding="utf-8"))
    clip = json.loads(clip_path.read_text(encoding="utf-8"))
    summary["perceptual_metrics"] = {"lpips": {}, "clip": {}}
    for arm in ("rtn", "ptq"):
        values = [prompt["arms"][arm]["mean"] for prompt in lpips["prompts"]]
        summary["perceptual_metrics"]["lpips"][arm] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "values": values,
        }
    for arm in ("bf16", "rtn", "ptq"):
        values = [prompt["arms"][arm] for prompt in clip["prompts"]]
        entry = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "values": values,
        }
        if arm != "bf16":
            entry["mean_delta_from_bf16"] = statistics.fmean(
                prompt["arms"][arm] - prompt["arms"]["bf16"]
                for prompt in clip["prompts"]
            )
        summary["perceptual_metrics"]["clip"][arm] = entry
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=("lpips", "clip"), required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-cache", type=Path, default=Path("/hf_cache/vbench"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.metric == "lpips":
        payload = compute_lpips(args.report, args.device)
    else:
        payload = compute_clip(args.report, args.device, args.clip_cache)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
