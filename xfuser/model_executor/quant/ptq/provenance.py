"""Small, dependency-free provenance helpers for PTQ jobs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


def _git(path: Path, *args: str) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ("git", "-c", f"safe.directory={path}", "-C", str(path), *args),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def repository_state(path: str | Path) -> dict | None:
    path = Path(path)
    commit = _git(path, "rev-parse", "HEAD")
    if commit is None:
        return None
    diff = subprocess.run(
        (
            "git",
            "-c",
            f"safe.directory={path}",
            "-C",
            str(path),
            "diff",
            "--binary",
            "HEAD",
        ),
        check=False,
        capture_output=True,
    )
    return {
        "path": str(path),
        "commit": commit,
        "dirty": bool(diff.stdout),
        "diff_sha256": hashlib.sha256(diff.stdout).hexdigest(),
    }


def runtime_manifest() -> dict:
    selected_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("XFUSER_", "AITER_", "HIP_", "BRR_"))
    }
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": dict(sorted(selected_environment.items())),
        "gpu": (
            {
                "name": torch.cuda.get_device_name(),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            }
            if torch.cuda.is_available()
            else None
        ),
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "diffusers": importlib.metadata.version("diffusers"),
        },
        "repositories": {
            "xdit": repository_state("/app/xDiT"),
            "harness": repository_state("/workspace/wan-harness"),
        },
    }
