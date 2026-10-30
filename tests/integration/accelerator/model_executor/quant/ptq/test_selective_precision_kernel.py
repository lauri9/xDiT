"""A6W4 promotion of a load-time packed MXFP4 linear (MI355X)."""
import pytest
import torch

from xfuser.model_executor.quant.ptq.selective_precision import install_selective_precision

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


def test_promoted_linear_matches_direct_a6w4_and_beats_mxfp4():
    from xfuser.model_executor.layers.mixed_mxfp_linear import xFuserA6W4Linear
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    torch.manual_seed(0)
    weight = (torch.randn(256, 512, device="cuda") / 20).bfloat16()
    bias = (0.1 * torch.randn(256, device="cuda")).bfloat16()
    x = (torch.randn(300, 512, device="cuda") * torch.linspace(0.1, 4, 512, device="cuda")).bfloat16()
    holder = torch.nn.Module()
    block = torch.nn.Module()
    block.ffn = torch.nn.Module()
    linear = xFuserMXFP4Linear(512, 256, device="cuda", dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    linear.bias.data.copy_(bias)
    linear._quantize_weights()
    block.ffn.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), linear])
    holder.blocks = torch.nn.ModuleList([block])
    mxfp4 = linear(x)
    source = {"blocks.0.ffn.net.2.weight": weight, "blocks.0.ffn.net.2.bias": bias}
    install_selective_precision(holder, ["ffn_out"], weight_source=source.__getitem__)
    promoted = holder.blocks[0].ffn.net[2]
    direct = xFuserA6W4Linear(512, 256, device="cuda", dtype=torch.bfloat16)
    direct.load_and_quantize_weights(weight, bias)
    torch.testing.assert_close(promoted(x), direct(x), atol=0, rtol=0)
    exact = torch.nn.functional.linear(x.float(), weight.float(), bias.float())
    error = lambda y: float((y.float() - exact).square().sum() / exact.square().sum())
    assert error(promoted(x)) < error(mxfp4)
