import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.observers import (
    HessianAccumulator,
    WanHessianObservers,
)


class Attention(torch.nn.Module):
    def __init__(self, hidden=4, context=3):
        super().__init__()
        self.fused_projections = False
        self.to_q = torch.nn.Linear(hidden, hidden, bias=False)
        self.to_k = torch.nn.Linear(context, hidden, bias=False)
        self.to_v = torch.nn.Linear(context, hidden, bias=False)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden, bias=False)])


class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 6, bias=False)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention(hidden=4, context=4)
        self.attn2 = Attention(hidden=4, context=3)
        self.ffn = torch.nn.Module()
        self.ffn.net = torch.nn.ModuleList(
            [Projection(), torch.nn.GELU(), torch.nn.Linear(6, 4, bias=False)]
        )


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block()])


def test_hessian_accumulator_matches_masked_xtx_and_stats():
    inputs = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
    )
    mask = torch.tensor([[True, False], [True, False]])
    accumulator = HessianAccumulator(2, device="cpu")

    accumulator.add(inputs, mask)

    selected = inputs.reshape(-1, 2)[mask.reshape(-1)]
    torch.testing.assert_close(accumulator.hessian, selected.T @ selected)
    torch.testing.assert_close(accumulator.absmax, selected.abs().amax(dim=0))
    torch.testing.assert_close(
        accumulator.rms,
        (selected.square().mean(dim=0)).sqrt(),
    )
    assert accumulator.token_count == 2


def test_split_hessians_merge_to_the_full_hessian():
    values = torch.randn(20, 4, generator=torch.Generator().manual_seed(3))
    full = HessianAccumulator(4, device="cpu")
    left = HessianAccumulator(4, device="cpu")
    right = HessianAccumulator(4, device="cpu")

    full.add(values)
    left.add(values[:7])
    right.add(values[7:])
    left.merge_(right)

    torch.testing.assert_close(left.hessian, full.hessian)
    torch.testing.assert_close(left.absmax, full.absmax)
    torch.testing.assert_close(left.rms, full.rms)
    assert left.token_count == full.token_count


def test_sampling_is_deterministic_under_a_seed():
    values = torch.randn(100, 4, generator=torch.Generator().manual_seed(9))
    first = HessianAccumulator(
        4,
        device="cpu",
        token_sample_rate=0.25,
        tokens_per_forward=20,
        seed=17,
    )
    second = HessianAccumulator(
        4,
        device="cpu",
        token_sample_rate=0.25,
        tokens_per_forward=20,
        seed=17,
    )

    first.add(values)
    second.add(values)

    assert first.token_count == 20
    torch.testing.assert_close(first.hessian, second.hessian)


def test_wan_observers_share_seven_unique_inputs_per_block():
    transformer = Transformer()
    observers = WanHessianObservers(transformer, device="cpu")
    block = transformer.blocks[0]
    hidden = torch.randn(2, 5, 4)
    context = torch.randn(2, 7, 3)

    block.attn1.to_q(hidden)
    block.attn1.to_k(hidden)
    block.attn1.to_v(hidden)
    block.attn1.to_out[0](hidden)
    block.attn2.to_q(hidden)
    block.attn2.to_k(context)
    block.attn2.to_v(context)
    block.attn2.to_out[0](hidden)
    projected = block.ffn.net[0].proj(hidden)
    block.ffn.net[2](projected)

    assert len(observers.groups) == 7
    assert (
        observers.linear_to_group["blocks.0.attn1.to_q"]
        == observers.linear_to_group["blocks.0.attn1.to_k"]
        == observers.linear_to_group["blocks.0.attn1.to_v"]
    )
    assert observers.groups["blocks.0.self_qkv"].token_count == 10
    assert observers.groups["blocks.0.cross_kv"].token_count == 14
    observers.remove()


def test_fused_attention_is_rejected_before_hooks_are_installed():
    transformer = Transformer()
    transformer.blocks[0].attn1.fused_projections = True

    with pytest.raises(ValueError, match="fused projections"):
        WanHessianObservers(transformer, device="cpu")
