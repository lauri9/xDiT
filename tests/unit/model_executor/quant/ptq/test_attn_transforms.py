import json

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from torch import nn

from xfuser.model_executor.quant.ptq.attn_transforms import (
    AttentionTransform,
    apply_attention_artifact,
    fit_t1,
    fit_t2,
    fit_t3,
    fold_attention_state_dict,
    load_attention_artifact,
    write_attention_artifact,
)


REVISION = "5be7df9619b54f4e2667b2755bc6a756675b5cd7"


def _stats(values):
    flat = values.reshape(-1, *values.shape[-2:]).double()
    return {
        "sum": flat.sum(0),
        "sum_squares": flat.square().sum(0),
        "absmax": flat.abs().amax(0).float(),
        "token_count": flat.shape[0],
    }


def test_t1_softmax_invariance_and_moment_fits():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 2, 4)
    k = torch.randn(2, 5, 2, 4)
    v = torch.randn(2, 5, 2, 4)
    k_shift = fit_t1(_stats(k))
    v_shift = fit_t2(_stats(v))

    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    actual = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        (k - k_shift).transpose(1, 2),
        v.transpose(1, 2),
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(k_shift, k.mean((0, 1)))
    torch.testing.assert_close(v_shift, v.mean((0, 1)))


def test_t2_state_dict_fold_preserves_projected_attention_output():
    torch.manual_seed(1)
    heads, dim, model_dim = 2, 4, 8
    prefix = "blocks.0.attn1"
    state = {
        f"{prefix}.to_v.bias": torch.randn(model_dim),
        f"{prefix}.to_out.0.weight": torch.randn(model_dim, model_dim),
        f"{prefix}.to_out.0.bias": torch.randn(model_dim),
        f"{prefix}.norm_q.weight": torch.randn(model_dim),
        f"{prefix}.norm_k.weight": torch.randn(model_dim),
    }
    transform = AttentionTransform(
        k_shift=torch.zeros(heads, dim),
        v_shift=torch.randn(heads, dim),
        scale=torch.ones(heads, dim),
    )
    folded = fold_attention_state_dict(state, prefix, transform)
    probabilities = torch.softmax(torch.randn(3, 5), dim=-1)
    projected_v = torch.randn(5, model_dim) + state[f"{prefix}.to_v.bias"]
    expected = (
        probabilities @ projected_v
    ) @ state[f"{prefix}.to_out.0.weight"].T + state[f"{prefix}.to_out.0.bias"]
    centered_v = (
        projected_v
        - state[f"{prefix}.to_v.bias"]
        + folded[f"{prefix}.to_v.bias"]
    )
    actual = (
        probabilities @ centered_v
    ) @ folded[f"{prefix}.to_out.0.weight"].T + folded[
        f"{prefix}.to_out.0.bias"
    ]

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_t3_pair_tied_scale_commutes_with_rope_and_preserves_logits():
    torch.manual_seed(2)
    q = torch.randn(2, 7, 3, 8)
    k = torch.randn(2, 7, 3, 8)
    scale = fit_t3(_stats(q), _stats(k), alpha=0.5)
    torch.testing.assert_close(scale[..., 0::2], scale[..., 1::2])
    theta = torch.randn(1, 7, 1, 4)
    cos, sin = theta.cos(), theta.sin()

    def rope(x):
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), dim=-1
        ).flatten(-2)

    torch.testing.assert_close(rope(q / scale), rope(q) / scale)
    torch.testing.assert_close(rope(k * scale), rope(k) * scale)
    expected = torch.einsum("bshd,bthd->bhst", rope(q), rope(k))
    actual = torch.einsum(
        "bshd,bthd->bhst", rope(q / scale), rope(k * scale)
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_v = nn.Linear(4, 4)
        self.to_out = nn.Sequential(nn.Linear(4, 4), nn.Identity())
        self.norm_q = nn.LayerNorm(4, elementwise_affine=True)
        self.norm_k = nn.LayerNorm(4, elementwise_affine=True)
        self.register_buffer("k_shift", torch.zeros(2, 2), persistent=False)


class _Component(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn1 = _Attention()
        self.blocks = nn.ModuleList([block])


def test_attention_artifact_round_trip_validation_and_application(tmp_path):
    root = tmp_path / "artifact"
    name = "transformer.blocks.0.attn1"
    keys = {
        "k_shift": f"{name}.k_shift",
        "v_bias_delta": f"{name}.v_bias_delta",
        "out_bias_delta": f"{name}.out_bias_delta",
        "scale": f"{name}.scale",
    }
    tensors = {
        keys["k_shift"]: torch.full((2, 2), 0.25),
        keys["v_bias_delta"]: torch.full((2, 2), -0.5),
        keys["out_bias_delta"]: torch.arange(4.0),
        keys["scale"]: torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
    }
    write_attention_artifact(
        root,
        artifact_id="attn-1",
        source_model="model",
        source_revision=REVISION,
        calib_id="calib-1",
        fit_config={"alpha": 0.5},
        tensors=tensors,
        blocks={name: {"shape": [2, 2], "tensors": keys}},
    )
    artifact = load_attention_artifact(
        root, expected_source_revision=REVISION
    )
    component = _Component()
    attention = component.get_submodule("blocks.0.attn1")
    old_v_bias = attention.to_v.bias.detach().clone()
    old_out_bias = attention.to_out[0].bias.detach().clone()
    apply_attention_artifact({"transformer": component}, artifact)

    torch.testing.assert_close(attention.k_shift, tensors[keys["k_shift"]])
    torch.testing.assert_close(
        attention.to_v.bias, old_v_bias + tensors[keys["v_bias_delta"]].flatten()
    )
    torch.testing.assert_close(
        attention.to_out[0].bias, old_out_bias + tensors[keys["out_bias_delta"]]
    )
    torch.testing.assert_close(
        attention.norm_q.weight, 1 / tensors[keys["scale"]].flatten()
    )
    torch.testing.assert_close(
        attention.norm_k.weight, tensors[keys["scale"]].flatten()
    )
    with pytest.raises(RuntimeError, match="already applied"):
        apply_attention_artifact({"transformer": component}, artifact)

    with pytest.raises(ValueError, match="revision mismatch"):
        load_attention_artifact(root, expected_source_revision="wrong")
    tensor_path = root / "attention_transforms.safetensors"
    original = tensor_path.read_bytes()
    corrupted = bytearray(original)
    corrupted[-1] ^= 1
    tensor_path.write_bytes(corrupted)
    with pytest.raises(ValueError, match="tensor hash"):
        load_attention_artifact(root, expected_source_revision=REVISION)
    tensor_path.write_bytes(original)

    config = json.loads((root / "attn_config.json").read_text())
    config["fit_config"]["alpha"] = 0.25
    (root / "attn_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="metadata hash"):
        load_attention_artifact(root, expected_source_revision=REVISION)
