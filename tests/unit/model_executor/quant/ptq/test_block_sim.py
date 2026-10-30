import pytest
import torch

from xfuser.model_executor.quant.ptq.block_sim import (
    BlockConfig, GemmSetting, RotationPlan, UpdateError, WanBlockSim, fake_quant_mxfp4,
    parameter_names, prepare_block,
)
from xfuser.model_executor.quant.ptq.mxfp4_ref import dequantize_mxfp4, quantize_mxfp4
from xfuser.model_executor.quant.ptq.rotations import HEAD_DIM, Rotation, make_rotation

HEADS, DIM, FFN, SEQ, TEXT = 2, 256, 512, 64, 16


def _params(seed=0):
    generator = torch.Generator().manual_seed(seed)
    shapes = {
        "attn1.to_q": (DIM, DIM), "attn1.to_k": (DIM, DIM), "attn1.to_v": (DIM, DIM), "attn1.to_out.0": (DIM, DIM),
        "attn2.to_q": (DIM, DIM), "attn2.to_k": (DIM, DIM), "attn2.to_v": (DIM, DIM), "attn2.to_out.0": (DIM, DIM),
        "ffn.net.0.proj": (FFN, DIM), "ffn.net.2": (DIM, FFN),
    }
    params = {}
    for name in parameter_names():
        if name.endswith(".weight") and name[:-7] in shapes:
            shape = shapes[name[:-7]]
            params[name] = torch.randn(*shape, generator=generator) / shape[1] ** 0.5
        elif name.endswith(".bias") and name[:-5] in shapes:
            params[name] = 0.1 * torch.randn(shapes[name[:-5]][0], generator=generator)
        elif name == "scale_shift_table":
            params[name] = 0.1 * torch.randn(1, 6, DIM, generator=generator)
        elif name == "norm2.bias":
            params[name] = 0.1 * torch.randn(DIM, generator=generator)
        else:
            params[name] = 1 + 0.1 * torch.randn(DIM, generator=generator)
    return params


def _inputs(dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, SEQ, DIM, generator=generator).to(dtype)
    encoder = torch.randn(1, TEXT, DIM, generator=generator).to(dtype)
    temb = 0.1 * torch.randn(1, 6, DIM, generator=generator)
    angles = torch.randn(1, SEQ, 1, HEAD_DIM // 2, generator=generator).repeat_interleave(2, -1)
    return hidden, encoder, temb, (angles.cos(), angles.sin())


def _full_plan():
    return RotationPlan(
        qk=make_rotation(HEAD_DIM, "wht", sign_seed=3, sign_pairs=True), qk_mode="fold_signs",
        self_vo=make_rotation(HEAD_DIM, "haar", seed=4), cross_vo=make_rotation(HEAD_DIM, "wht", block=32),
        online={"self_qkv": make_rotation(DIM, "wht", block=128), "cross_q": make_rotation(DIM, "dct"),
                "ffn_in": make_rotation(DIM, "haar", block=64, seed=1), "ffn_out": make_rotation(FFN, "wht", block=32),
                "cross_out": make_rotation(DIM, "wht", block=32)},
        ffn_perm=torch.randperm(FFN, generator=torch.Generator().manual_seed(5)),
        cross_kv_input=make_rotation(DIM, "haar", seed=6),
    )


def test_rotation_plan_preserves_the_unquantized_block():
    params, inputs = _params(), _inputs()
    config = BlockConfig.uniform("bf16", "bf16", attention_impl="reference")
    base = WanBlockSim(prepare_block(params, None), config, heads=HEADS).forward(*inputs)
    rotated = WanBlockSim(prepare_block(params, _full_plan()), config, heads=HEADS).forward(*inputs)
    torch.testing.assert_close(rotated, base, atol=2e-4, rtol=2e-4)


def test_prerotation_plan_is_inert_without_quantized_attention():
    params, inputs = _params(), _inputs()
    config = BlockConfig.uniform("bf16", "bf16", attention_impl="reference")
    plan = RotationPlan(qk=make_rotation(HEAD_DIM, "haar", seed=2))
    torch.testing.assert_close(WanBlockSim(prepare_block(params, plan), config, heads=HEADS).forward(*inputs),
                               WanBlockSim(prepare_block(params, None), config, heads=HEADS).forward(*inputs))


def test_sign_fold_equals_sign_prerotation_with_quantized_attention():
    params, inputs = _params(), _inputs(torch.bfloat16)
    config = BlockConfig.uniform("emulated", "mxfp4", attention_impl="reference")
    rotation = make_rotation(HEAD_DIM, "wht", sign_seed=9, sign_pairs=True)
    folded = WanBlockSim(prepare_block(params, RotationPlan(qk=rotation, qk_mode="fold_signs")), config, heads=HEADS)
    pre = WanBlockSim(prepare_block(params, RotationPlan(qk=rotation)), config, heads=HEADS)
    out_folded, out_pre = folded.forward(*inputs), pre.forward(*inputs)
    base = WanBlockSim(prepare_block(params, None), config, heads=HEADS).forward(*inputs)
    assert (out_folded.float() - out_pre.float()).abs().max() < 0.25 * (out_folded.float() - base.float()).abs().max()


def test_identity_prerotation_turns_off_the_kernel_rotation():
    params, inputs = _params(), _inputs(torch.bfloat16)
    config = BlockConfig.uniform("bf16", "mxfp4", attention_impl="reference")
    taps_none, taps_prod = {}, {}
    WanBlockSim(prepare_block(params, RotationPlan(qk=Rotation.identity(HEAD_DIM))), config, heads=HEADS).forward(
        *inputs, taps=taps_none)
    WanBlockSim(prepare_block(params, None), config, heads=HEADS).forward(*inputs, taps=taps_prod)
    assert not torch.equal(taps_none["self_attn"], taps_prod["self_attn"])


def test_emulated_gemm_modes_and_overrides():
    params, inputs = _params(), _inputs(torch.bfloat16)
    weight_only = BlockConfig({g: GemmSetting("emulated", quant_act=False) for g in
                               ("self_qkv", "self_out", "cross_q", "cross_kv", "cross_out", "ffn_in", "ffn_out")},
                              "bf16", attention_impl="reference")
    sim = WanBlockSim(prepare_block(params, None), weight_only, heads=HEADS)
    expected = dequantize_mxfp4(*quantize_mxfp4(params["ffn.net.2.weight"].bfloat16()))
    torch.testing.assert_close(sim._fq_weights["ffn.net.2"], expected)
    override = {"ffn.net.2": torch.zeros(DIM, FFN)}
    with_override = WanBlockSim(prepare_block(params, None), weight_only, heads=HEADS, weight_overrides=override)
    assert torch.equal(with_override._fq_weights["ffn.net.2"], override["ffn.net.2"])
    sim.forward(*inputs)


def test_fake_quant_chunking_matches_unchunked():
    x = torch.randn(70, 64)
    torch.testing.assert_close(fake_quant_mxfp4(x, chunk_rows=16), dequantize_mxfp4(*quantize_mxfp4(x)))


def test_update_error_cfg_combination():
    accumulator = UpdateError()
    x = [torch.zeros(4), torch.zeros(4)]
    ref = [torch.ones(4), 0.5 * torch.ones(4)]
    cand = [torch.ones(4) * 1.1, 0.5 * torch.ones(4)]
    record = accumulator.add_pair(x, ref, cand, guidance=3.0)
    # cfg ref = 3*1 - 2*0.5 = 2 per element; cfg error = 3*0.1
    assert record["cfg_update_rel_mse"] == pytest.approx(0.09 / 4)
    assert accumulator.report()["pairs"] == 1
