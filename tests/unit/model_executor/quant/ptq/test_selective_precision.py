import pytest
import torch

from xfuser.model_executor.quant.ptq.selective_precision import install_selective_precision


class FakeHigh(torch.nn.Module):
    """Stand-in for xFuserA6W4Linear's construction/packing contract."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        assert device == "meta" and dtype == torch.bfloat16
        self.in_features, self.out_features = in_features, out_features
        self.has_bias = bias

    def load_and_quantize_weights(self, weights, bias=None, *, device=None):
        self.packed = weights.clone()
        self.bias = None if bias is None else torch.nn.Parameter(bias.clone())
        self.device = device


def _holder():
    block = torch.nn.Module()
    block.attn1 = torch.nn.Module()
    block.attn1.to_out = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Dropout()])
    block.ffn = torch.nn.Module()
    block.ffn.net = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Dropout(), torch.nn.Linear(16, 8)])
    block.ffn.net[0].proj = torch.nn.Linear(8, 16)
    holder = torch.nn.Module()
    holder.blocks = torch.nn.ModuleList([block, block.__class__()])
    return holder


def test_replaces_only_selected_groups_and_blocks_with_checkpoint_weights():
    holder = _holder()
    source = {f"blocks.0.{n}.{k}": torch.full(s, v, dtype=torch.float32)
              for n, shapes, v in (("ffn.net.2", ((8, 16), (8,)), 2.0), ("attn1.to_out.0", ((8, 8), (8,)), 3.0))
              for k, s in zip(("weight", "bias"), shapes)}
    replaced = install_selective_precision(holder, ["ffn_out", "self_out"], weight_source=source.__getitem__,
                                           blocks=[0], factory=FakeHigh)
    assert replaced == ["blocks.0.ffn.net.2", "blocks.0.attn1.to_out.0"]
    ffn = holder.blocks[0].ffn.net[2]
    assert isinstance(ffn, FakeHigh) and ffn.packed.dtype == torch.bfloat16 and float(ffn.packed[0, 0]) == 2.0
    assert isinstance(holder.blocks[0].attn1.to_out[0], FakeHigh)
    assert isinstance(holder.blocks[0].ffn.net[0].proj, torch.nn.Linear)
    with pytest.raises(RuntimeError, match="already"):
        install_selective_precision(holder, ["ffn_out"], weight_source=source.__getitem__, blocks=[0],
                                    factory=FakeHigh)
    with pytest.raises(ValueError, match="unknown"):
        install_selective_precision(holder, ["ffn_typo"], weight_source=source.__getitem__, factory=FakeHigh)


class FakeMXFP4(torch.nn.Module):
    """Stand-in for a loaded xFuserMXFP4Linear (packed weight, live bias)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_shuffle = torch.nn.Parameter(torch.zeros(out_features, in_features // 2), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.full((out_features,), 7.0))
        self.input_rotation = None
        self.loaded = None

    def load_and_quantize_weights(self, weights, bias=None):
        assert bias is None
        self.loaded = weights.clone()


def test_repack_mxfp4_weights_reloads_in_place_and_keeps_bias():
    from xfuser.model_executor.quant.ptq.selective_precision import repack_mxfp4_weights

    holder = _holder()
    holder.blocks[0].ffn.net[2] = FakeMXFP4(16, 8)
    layer = holder.blocks[0].ffn.net[2]
    source = {"blocks.0.ffn.net.2.weight": torch.full((8, 16), 0.5)}
    names = repack_mxfp4_weights(holder, ["ffn_out"], weight_source=source.__getitem__, blocks=[0],
                                 layer_type=FakeMXFP4)
    assert names == ["blocks.0.ffn.net.2"]
    assert holder.blocks[0].ffn.net[2] is layer
    assert layer.loaded.dtype == torch.bfloat16 and float(layer.loaded[0, 0]) == 0.5
    assert float(layer.bias[0]) == 7.0
    with pytest.raises(RuntimeError, match="not FakeMXFP4"):
        repack_mxfp4_weights(holder, ["self_out"], weight_source=source.__getitem__, blocks=[0],
                             layer_type=FakeMXFP4)
    layer.input_rotation = object()
    with pytest.raises(RuntimeError, match="input rotation"):
        repack_mxfp4_weights(holder, ["ffn_out"], weight_source=source.__getitem__, blocks=[0],
                             layer_type=FakeMXFP4)
