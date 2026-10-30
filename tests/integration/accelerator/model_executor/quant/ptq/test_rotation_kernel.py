"""Kernel-side checks for the rotation experiments (pre-rotation, rotated GEMM,
block simulator). Only the claims the ablation relies on are tested."""
import pytest
import torch

from xfuser.model_executor.quant.ptq.attn_mxfp4_ref import mxfp4_attention_ref
from xfuser.model_executor.quant.ptq.block_sim import (
    BlockConfig, KernelGemm, RotationPlan, WanBlockSim, parameter_names, prepare_block,
)
from xfuser.model_executor.quant.ptq.rotations import (
    HEAD_DIM, Rotation, apply_head_matrix, fold_input, make_rotation, qk_prerotation,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


def _rel(a, b):
    return float((a.float() - b.float()).square().sum() / b.float().square().sum())


@pytest.mark.parametrize("build", [
    lambda: Rotation.identity(HEAD_DIM),
    lambda: make_rotation(HEAD_DIM, "wht", sign_seed=5, sign_pairs=True),
    lambda: make_rotation(HEAD_DIM, "wht", block=32),
    lambda: make_rotation(HEAD_DIM, "haar", seed=8),
])
def test_mha_v4_prerotation_matches_emulator(build):
    mha = pytest.importorskip("aiter.ops.mha_v4")
    generator = torch.Generator(device="cuda").manual_seed(7)
    q, k, v = (torch.randn(1, 208, 2, HEAD_DIM, device="cuda", generator=generator).bfloat16() for _ in range(3))
    pre = qk_prerotation(build(), "cuda")
    qp, kp = apply_head_matrix(q, pre).bfloat16(), apply_head_matrix(k, pre).bfloat16()
    fmt = mha.AttentionFormat.MXFP4
    kernel = mha.mha_v4(qp, kp, v, fmt, fmt, fmt)
    assert _rel(kernel, mxfp4_attention_ref(qp, kp, v)) < 1e-5


def test_kernel_gemm_matches_mxfp4_linear_and_input_rotation():
    from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear

    torch.manual_seed(0)
    x = torch.randn(300, 512, device="cuda").bfloat16()
    weight = (torch.randn(256, 512, device="cuda") / 20).bfloat16()
    linear = xFuserMXFP4Linear(512, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    linear.weight.data.copy_(weight)
    torch.testing.assert_close(KernelGemm(weight)(x), linear(x), atol=0, rtol=0)

    rotation = make_rotation(512, "wht", block=128, sign_seed=1)
    folded = fold_input(weight.float(), rotation).bfloat16()
    rotated = xFuserMXFP4Linear(512, 256, bias=False, device="cuda", dtype=torch.bfloat16)
    rotated.weight.data.copy_(folded)
    rotated.input_rotation = rotation
    torch.testing.assert_close(rotated(x), KernelGemm(folded)(rotation.apply(x).bfloat16()), atol=0, rtol=0)


def test_block_sim_kernel_matches_emulated():
    torch.manual_seed(1)
    heads, dim, ffn, seq = 2, 256, 512, 208
    params = {}
    for name in parameter_names():
        stem = name.rsplit(".", 1)[0]
        out_dim = ffn if stem == "ffn.net.0.proj" else dim
        in_dim = ffn if stem == "ffn.net.2" else dim
        if name.endswith(".weight") and "norm" not in name:
            params[name] = (torch.randn(out_dim, in_dim, device="cuda") / in_dim**0.5).bfloat16()
        elif name == "scale_shift_table":
            params[name] = (0.1 * torch.randn(1, 6, dim, device="cuda")).bfloat16()
        elif name.endswith(".bias") and "norm" not in name:
            params[name] = (0.1 * torch.randn(out_dim, device="cuda")).bfloat16()
        else:
            params[name] = (1 + 0.1 * torch.randn(dim, device="cuda")).bfloat16()
    hidden = torch.randn(1, seq, dim, device="cuda").bfloat16()
    encoder = torch.randn(1, 32, dim, device="cuda").bfloat16()
    temb = 0.1 * torch.randn(1, 6, dim, device="cuda")
    angles = torch.randn(1, seq, 1, HEAD_DIM // 2, device="cuda").repeat_interleave(2, -1)
    inputs = (hidden, encoder, temb, (angles.cos(), angles.sin()))
    plan = RotationPlan(online={"ffn_out": make_rotation(ffn, "wht", block=32)},
                        qk=make_rotation(HEAD_DIM, "wht", sign_seed=2, sign_pairs=True))
    outputs = {}
    for mode in ("bf16", "emulated", "kernel"):
        attention = "bf16" if mode == "bf16" else "mxfp4"
        sim = WanBlockSim(prepare_block(params, plan), BlockConfig.uniform(mode, attention), heads=heads)
        outputs[mode] = sim.forward(*inputs) - hidden
    quant_error = _rel(outputs["kernel"], outputs["bf16"])
    assert quant_error > 1e-4
    assert _rel(outputs["emulated"], outputs["kernel"]) < 0.02 * quant_error
