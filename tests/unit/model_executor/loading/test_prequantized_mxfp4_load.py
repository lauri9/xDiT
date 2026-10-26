from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear
from xfuser.model_executor.models.runner_models.loading.checkpoint import (
    CheckpointManifest,
)
from xfuser.model_executor.models.runner_models.loading.meta_load import (
    _BlockwiseDiskFiller,
)
from xfuser.model_executor.quant.ptq.prequantized_load import (
    PackedLinearSpec,
    compose_prequantized_manifest,
    install_prequantized_mxfp4_layout,
)


def _spec():
    return PackedLinearSpec(
        module_name="blocks.0.proj",
        packed_key="blocks.0.proj.weight_packed",
        scale_key="blocks.0.proj.weight_scale",
        packed_shape=(33, 32),
        shuffled_scale_shape=(256, 8),
    )


def test_meta_linear_is_replaced_before_fill_and_keeps_bf16_bias():
    with torch.device("meta"):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList(
            [torch.nn.ModuleDict({"proj": torch.nn.Linear(64, 33, bias=True)})]
        )
    model.to(dtype=torch.bfloat16)

    install_prequantized_mxfp4_layout(model, [_spec()])

    linear = model.blocks[0].proj
    assert isinstance(linear, xFuserMXFP4Linear)
    assert linear.weight is None
    assert linear.weight_shuffle.shape == (33, 32)
    assert linear.weight_scale.shape == (256, 8)
    assert linear.bias.dtype == torch.bfloat16
    assert linear.bias.device.type == "meta"


def test_manifest_overlays_packed_state_without_mapping_bf16_weight():
    base = CheckpointManifest(
        weight_map={
            "blocks.0.proj.weight": "bf16-1.safetensors",
            "blocks.0.proj.bias": "bf16-1.safetensors",
            "norm.weight": "bf16-2.safetensors",
        }
    )

    manifest = compose_prequantized_manifest(
        base,
        artifact_path="artifact.safetensors",
        specs=[_spec()],
        weight_builder=lambda value: value + 1,
        scale_builder=lambda value: value + 2,
    )

    assert "blocks.0.proj.weight" not in manifest.weight_map
    assert manifest.weight_map["blocks.0.proj.bias"] == "bf16-1.safetensors"
    assert manifest.weight_map["blocks.0.proj.weight_shuffle"] == "artifact.safetensors"
    assert manifest.weight_map["blocks.0.proj.weight_scale"] == "artifact.safetensors"
    assert manifest.sparse_paths == {
        "bf16-1.safetensors",
        "bf16-2.safetensors",
    }

    artifact = {
        _spec().packed_key: torch.tensor([3], dtype=torch.uint8),
        _spec().scale_key: torch.tensor([5], dtype=torch.uint8),
    }
    filler = object.__new__(_BlockwiseDiskFiller)
    filler.weight_map = manifest.weight_map
    filler.checkpoint_keys = manifest.checkpoint_keys
    filler.derived = manifest.derived
    filler._handle = lambda _path: SimpleNamespace(get_tensor=artifact.__getitem__)

    packed = filler._tensor_for("blocks.0.proj.weight_shuffle", "artifact.safetensors")
    scale = filler._tensor_for("blocks.0.proj.weight_scale", "artifact.safetensors")

    assert packed.item() == 4
    assert scale.item() == 7


def test_manifest_rejects_artifact_target_absent_from_bf16_checkpoint():
    with pytest.raises(ValueError, match="has no target weight"):
        compose_prequantized_manifest(
            CheckpointManifest(weight_map={"norm.weight": "bf16.safetensors"}),
            artifact_path="artifact.safetensors",
            specs=[_spec()],
        )
