from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from xfuser.model_executor.quant.ptq.attn_observers import (
    ChannelStatsAccumulator,
    WanAttentionStatsObservers,
)


def test_channel_stats_mask_valid_kv_len():
    values = torch.tensor(
        [
            [
                [[1.0, -2.0], [3.0, 4.0]],
                [[5.0, 6.0], [-7.0, 8.0]],
                [[100.0, 100.0], [100.0, 100.0]],
            ]
        ]
    )
    accumulator = ChannelStatsAccumulator(2, 2, device="cpu")

    accumulator.add(values, valid_kv_len=2)

    selected = values[:, :2].reshape(-1, 2, 2)
    torch.testing.assert_close(
        accumulator.sum, selected.sum(dim=0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        accumulator.sum_squares,
        selected.square().sum(dim=0, dtype=torch.float64),
    )
    torch.testing.assert_close(accumulator.absmax, selected.abs().amax(dim=0))
    assert accumulator.token_count == 2


def test_channel_stats_merge_matches_full_accumulation():
    values = torch.randn(2, 5, 3, 4, generator=torch.Generator().manual_seed(7))
    full = ChannelStatsAccumulator(3, 4, device="cpu")
    left = ChannelStatsAccumulator(3, 4, device="cpu")
    right = ChannelStatsAccumulator(3, 4, device="cpu")

    full.add(values)
    left.add(values[:, :2])
    right.add(values[:, 2:])
    left.merge_(right)

    torch.testing.assert_close(left.sum, full.sum)
    torch.testing.assert_close(left.sum_squares, full.sum_squares)
    torch.testing.assert_close(left.absmax, full.absmax)
    assert left.token_count == full.token_count


class _Attention(torch.nn.Module):
    def __init__(self, valid_kv_len):
        super().__init__()
        self.heads = 2
        self.to_q = torch.nn.Linear(4, 8, bias=False)
        self.processor = SimpleNamespace(
            qkv_observer=None,
            attention_kwargs={"valid_kv_len": valid_kv_len},
        )


class _Transformer(torch.nn.Module):
    def __init__(self, valid_kv_len=2):
        super().__init__()
        block = torch.nn.Module()
        block.attn1 = _Attention(valid_kv_len)
        self.blocks = torch.nn.ModuleList([block])


def test_wan_attention_observers_attach_mask_and_remove():
    transformer = _Transformer(valid_kv_len=2)
    observers = WanAttentionStatsObservers(transformer, device="cpu")
    callback = transformer.blocks[0].attn1.processor.qkv_observer
    values = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)

    callback(transformer.blocks[0].attn1, values, values + 1, values + 2)

    group = observers.groups["blocks.0.attn1"]
    assert group.query.token_count == 2
    assert group.key.token_count == 2
    assert group.value.token_count == 2
    torch.testing.assert_close(
        group.query.sum,
        values[:, :2].reshape(-1, 2, 4).sum(dim=0, dtype=torch.float64),
    )
    observers.remove()
    assert transformer.blocks[0].attn1.processor.qkv_observer is None


def test_wan_attention_observers_reject_an_existing_callback():
    transformer = _Transformer()
    transformer.blocks[0].attn1.processor.qkv_observer = lambda *_args: None

    with pytest.raises(RuntimeError, match="already has"):
        WanAttentionStatsObservers(transformer, device="cpu")
