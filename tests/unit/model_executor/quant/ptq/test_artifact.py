import json

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.artifact import (
    QuantizedLinear,
    load_artifact,
    write_artifact,
)


REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


def _linears():
    return {
        "transformer.blocks.0.attn1.to_q": QuantizedLinear(
            packed=torch.arange(64, dtype=torch.uint8).reshape(2, 32),
            scale=torch.tensor([[127, 128], [129, 130]], dtype=torch.uint8),
        ),
        "transformer_2.blocks.0.ffn.net.2": QuantizedLinear(
            packed=torch.full((2, 32), 3, dtype=torch.uint8),
            scale=torch.full((2, 2), 127, dtype=torch.uint8),
        ),
    }


def test_artifact_round_trip_and_component_specs(tmp_path):
    root = tmp_path / "artifact"
    written = write_artifact(
        root,
        artifact_id="artifact-1",
        source_model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        source_revision=REVISION,
        calib_id="calib-1",
        gptq_config={"percdamp": 0.01},
        linears=_linears(),
        max_shard_bytes=70,
    )

    loaded = load_artifact(
        root,
        expected_source_revision=REVISION,
        expected_layers={
            "transformer.blocks.0.attn1.to_q": (2, 64),
            "transformer_2.blocks.0.ffn.net.2": (2, 64),
        },
    )

    assert loaded.artifact_id == written.artifact_id
    assert len(loaded.config["shards"]) == 2
    specs = loaded.specs_for_component("transformer")
    assert [spec.module_name for spec in specs] == ["blocks.0.attn1.to_q"]
    assert specs[0].packed_shape == (2, 32)
    assert specs[0].shuffled_scale_shape == (256, 8)


def test_artifact_is_write_once(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()

    with pytest.raises(FileExistsError):
        write_artifact(
            root,
            artifact_id="artifact-1",
            source_model="model",
            source_revision=REVISION,
            calib_id="calib-1",
            gptq_config={},
            linears=_linears(),
        )


def test_artifact_rejects_revision_layer_and_hash_mismatches(tmp_path):
    root = tmp_path / "artifact"
    write_artifact(
        root,
        artifact_id="artifact-1",
        source_model="model",
        source_revision=REVISION,
        calib_id="calib-1",
        gptq_config={},
        linears=_linears(),
    )

    with pytest.raises(ValueError, match="revision mismatch"):
        load_artifact(root, expected_source_revision="wrong")
    with pytest.raises(ValueError, match="layer set or shape"):
        load_artifact(
            root,
            expected_source_revision=REVISION,
            expected_layers={"transformer.blocks.0.attn1.to_q": (2, 64)},
        )

    config = json.loads((root / "quant_config.json").read_text())
    shard = root / next(iter(config["shards"]))
    data = bytearray(shard.read_bytes())
    data[-1] ^= 1
    shard.write_bytes(data)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_artifact(root, expected_source_revision=REVISION)
