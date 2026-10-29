"""Importing the distributed stack must not import diffusers or AITER.

Workers spawn a fresh interpreter per rank. Those interpreters import
xfuser.core.distributed and then do a little collective work. Pipeline classes and
the AITER probe stay on the paths that actually build a model.
"""

import subprocess
import sys
import textwrap

import pytest


def _run(script: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(completed.stdout + completed.stderr)
    return completed.stdout


def test_importing_envs_does_not_import_diffusers_or_aiter():
    _run(
        """
        import sys
        import xfuser.envs
        assert "diffusers" not in sys.modules
        assert "aiter" not in sys.modules
        """
    )


def test_importing_distributed_does_not_import_pipelines_diffusers_or_aiter():
    _run(
        """
        import sys
        import xfuser.core.distributed
        leaked = [
            name
            for name in (
                "diffusers",
                "aiter",
                "yunchang",
                "xfuser.model_executor.pipelines",
                "xfuser.model_executor.pipelines.base_pipeline",
            )
            if name in sys.modules
        ]
        assert not leaked, leaked
        """
    )


def test_importing_args_does_not_import_diffusers_or_aiter():
    _run(
        """
        import sys
        from xfuser import xFuserArgs
        assert xFuserArgs is not None
        assert "diffusers" not in sys.modules
        assert "aiter" not in sys.modules
        """
    )


def test_xdit_help_does_not_import_models():
    _run(
        """
        import io
        import sys
        from xfuser.cli import main
        stdout = io.StringIO()
        sys.stdout = stdout
        main(["--help"])
        text = stdout.getvalue()
        assert text.startswith("usage: xdit"), text
        assert "--model" in text
        assert "--nproc_per_node" in text
        leaked = [
            name
            for name in (
                "diffusers",
                "aiter",
                "yunchang",
                "xfuser.runner",
                "xfuser.model_executor.pipelines",
                "xfuser.model_executor.models.runner_models.base_model",
            )
            if name in sys.modules
        ]
        assert not leaked, leaked
        """
    )


def test_runner_help_does_not_import_models():
    _run(
        """
        import io
        import sys
        sys.argv = ["xfuser.runner", "--help"]
        stdout = io.StringIO()
        sys.stdout = stdout
        import runpy
        try:
            runpy.run_module("xfuser.runner", run_name="__main__")
        except SystemExit as exc:
            assert exc.code in (None, 0)
        text = stdout.getvalue()
        assert text.startswith("usage: xfuser.runner"), text
        assert "--model" in text
        leaked = [
            name
            for name in (
                "diffusers",
                "aiter",
                "yunchang",
                "xfuser.model_executor.pipelines",
                "xfuser.model_executor.models.runner_models.base_model",
            )
            if name in sys.modules
        ]
        assert not leaked, leaked
        """
    )


def test_importing_a_pipeline_still_loads_diffusers_and_swaps_group_norm_on_rocm():
    _run(
        """
        import sys
        import torch
        import xfuser.envs as envs
        from xfuser import xFuserFluxPipeline
        assert xFuserFluxPipeline is not None
        assert "diffusers" in sys.modules
        if envs._is_hip():
            assert "aiter" in sys.modules
            assert torch.nn.GroupNorm.__module__ == "aiter.ops.groupnorm"
        """
    )
