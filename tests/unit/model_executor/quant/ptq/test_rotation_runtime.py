import pytest
import torch

from xfuser.model_executor.quant.ptq.rotation_runtime import install_online_rotations
from xfuser.model_executor.quant.ptq.rotations import make_rotation


class RotatableLinear(torch.nn.Linear):
    """CPU stand-in for xFuserMXFP4Linear's input-rotation contract."""

    def __init__(self, source: torch.nn.Linear):
        super().__init__(source.in_features, source.out_features, bias=source.bias is not None)
        self.load_state_dict(source.state_dict())
        self.input_rotation = None

    def forward(self, x):
        if self.input_rotation is not None:
            x = self.input_rotation.apply(x).to(x.dtype)
        return super().forward(x)


def _block():
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock

    torch.manual_seed(0)
    block = WanTransformerBlock(256, 512, 2, qk_norm="rms_norm_across_heads", cross_attn_norm=True, eps=1e-6)
    for name, module in list(block.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(module, child_name, RotatableLinear(child))
    return block.eval()


def test_installed_online_rotations_preserve_a_wan_block():
    pytest.importorskip("diffusers")
    block = _block()

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([block])

    holder = Holder()
    hidden, encoder, temb = torch.randn(1, 64, 256), torch.randn(1, 8, 256), 0.1 * torch.randn(1, 6, 256)
    angles = torch.randn(1, 64, 1, 64).repeat_interleave(2, -1)
    rotary = (angles.cos(), angles.sin())
    with torch.no_grad():
        expected = block(hidden, encoder, temb, rotary)
    changed = install_online_rotations(holder, {
        "self_qkv": make_rotation(256, "wht", block=128, sign_seed=0),
        "cross_q": make_rotation(256, "dct"),
        "ffn_out": make_rotation(512, "wht", block=32),
    })
    assert len(changed) == 5
    with torch.no_grad():
        torch.testing.assert_close(block(hidden, encoder, temb, rotary), expected, atol=5e-5, rtol=5e-5)
    with pytest.raises(RuntimeError, match="already has"):
        install_online_rotations(holder, {"ffn_out": make_rotation(512, "wht", block=32)})


def test_install_rejects_unrotatable_modules():
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inner = torch.nn.Module()
            inner.ffn = torch.nn.Module()
            inner.ffn.net = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(64, 8)])
            self.blocks = torch.nn.ModuleList([inner])

    with pytest.raises(TypeError, match="not a rotatable"):
        install_online_rotations(Holder(), {"ffn_out": make_rotation(64, "wht", block=32)})
