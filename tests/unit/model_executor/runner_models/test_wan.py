"""Runner contract for the Wan2.1 text-to-video 1.3B checkpoint.

Everything here builds the runner model from a parsed config only; nothing loads
weights or touches the network.
"""

import ast
from pathlib import Path

import pytest

# transformer/config.json of the 1.3B checkpoint.
WAN21_T2V_1_3B_HEADS = 12


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def _build(model_name="Wan2.1-T2V-1.3B", **config):
    from xfuser import xFuserArgs
    from xfuser.model_executor.models.runner_models.base_model import MODEL_REGISTRY

    return MODEL_REGISTRY[model_name](xFuserArgs(model=model_name, **config))


@pytest.mark.parametrize(
    "parallel",
    [
        {"ulysses_degree": 2},
        {"ulysses_degree": 3},
        {"ulysses_degree": 4},
        {"ring_degree": 2},
        {"ulysses_degree": 4, "ring_degree": 2},
    ],
)
def test_wan21_t2v_1_3b_accepts_ulysses_degrees_that_divide_its_heads(parallel):
    _build(**parallel)


def test_wan21_t2v_1_3b_rejects_ulysses_degree_that_splits_a_head():
    with pytest.raises(ValueError, match=rf"{WAN21_T2V_1_3B_HEADS} attention heads.*got 8.*--ring_degree"):
        _build(ulysses_degree=8)


def test_wan21_t2v_14b_keeps_accepting_ulysses_degree_8():
    _build("Wan2.1-T2V", ulysses_degree=8)


def test_all_standard_wan_pipeline_loads_forward_checkpoint_request_kwargs():
    """Root pipeline loads must use the same revision/cache request as their components."""

    source = Path(__file__).parents[4] / "xfuser/model_executor/models/runner_models/wan.py"
    tree = ast.parse(source.read_text())
    pipeline_owners = {
        "xFuserWanImageToVideoPipeline",
        "WanPipeline",
        "pipe_class",
        "WanVACEPipeline",
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in pipeline_owners
    ]

    assert len(calls) == 7
    for call in calls:
        expanded_kwargs = [keyword.value for keyword in call.keywords if keyword.arg is None]
        assert any(
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "from_pretrained_kwargs"
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "request"
            and any(
                keyword.arg == "include_subfolder"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is False
                for keyword in value.keywords
            )
            for value in expanded_kwargs
        )
