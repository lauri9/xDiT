"""Per-head, per-channel self-attention statistics for Wan calibration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ChannelStatsAccumulator:
    """Accumulate channel moments for tensors laid out as [B, S, H, D]."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        *,
        device: torch.device | str,
        chunk_tokens: int = 4096,
    ) -> None:
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("attention dimensions must be positive")
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.chunk_tokens = chunk_tokens
        shape = (num_heads, head_dim)
        self.sum = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self.sum_squares = torch.zeros(
            shape, dtype=torch.float64, device=self.device
        )
        self.absmax = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self.token_count = 0

    @torch.no_grad()
    def add(
        self,
        values: torch.Tensor,
        *,
        valid_kv_len: int | None = None,
    ) -> None:
        if values.ndim != 4 or values.shape[2:] != (
            self.num_heads,
            self.head_dim,
        ):
            raise ValueError(
                "attention observer expected [B, S, "
                f"{self.num_heads}, {self.head_dim}], got {tuple(values.shape)}"
            )
        sequence_length = values.shape[1]
        if valid_kv_len is not None:
            if isinstance(valid_kv_len, torch.Tensor):
                if valid_kv_len.numel() != 1:
                    raise ValueError("valid_kv_len must be a scalar")
                valid_kv_len = int(valid_kv_len.item())
            if not 0 <= valid_kv_len <= sequence_length:
                raise ValueError(
                    f"valid_kv_len={valid_kv_len} is outside [0, {sequence_length}]"
                )
            values = values[:, :valid_kv_len]

        flat = values.reshape(-1, self.num_heads, self.head_dim)
        for chunk in flat.split(self.chunk_tokens):
            chunk_f32 = chunk.to(device=self.device, dtype=torch.float32)
            self.sum.add_(chunk_f32.sum(dim=0, dtype=torch.float64))
            self.sum_squares.add_(
                chunk_f32.square().sum(dim=0, dtype=torch.float64)
            )
            torch.maximum(
                self.absmax,
                chunk_f32.abs().amax(dim=0),
                out=self.absmax,
            )
            self.token_count += chunk_f32.shape[0]

    def merge_(self, other: "ChannelStatsAccumulator") -> None:
        if (self.num_heads, self.head_dim) != (
            other.num_heads,
            other.head_dim,
        ):
            raise ValueError("cannot merge attention statistics with different shapes")
        self.sum.add_(other.sum.to(self.device))
        self.sum_squares.add_(other.sum_squares.to(self.device))
        torch.maximum(self.absmax, other.absmax.to(self.device), out=self.absmax)
        self.token_count += other.token_count

    def state_dict(self) -> dict[str, torch.Tensor | int]:
        return {
            "sum": self.sum.cpu(),
            "sum_squares": self.sum_squares.cpu(),
            "absmax": self.absmax.cpu(),
            "token_count": self.token_count,
        }


@dataclass
class AttentionBlockStats:
    query: ChannelStatsAccumulator
    key: ChannelStatsAccumulator
    value: ChannelStatsAccumulator

    def add(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        valid_kv_len: int | None,
    ) -> None:
        if query.shape != key.shape or key.shape != value.shape:
            raise ValueError("self-attention Q, K, and V shapes must match")
        for accumulator, tensor in (
            (self.query, query),
            (self.key, key),
            (self.value, value),
        ):
            accumulator.add(tensor, valid_kv_len=valid_kv_len)

    def merge_(self, other: "AttentionBlockStats") -> None:
        self.query.merge_(other.query)
        self.key.merge_(other.key)
        self.value.merge_(other.value)

    def state_dict(self) -> dict[str, dict[str, torch.Tensor | int]]:
        return {
            "q": self.query.state_dict(),
            "k": self.key.state_dict(),
            "v": self.value.state_dict(),
        }


class WanAttentionStatsObservers:
    """Install post-RoPE Q/K and V observers on Wan self-attention blocks.

    The callback is deliberately pre-transform: calibration always describes
    the BF16 source tensors, even if an attention artifact was accidentally
    supplied. Runtime diagnostics use the same unambiguous placement.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        device: torch.device | str,
        block_indices: tuple[int, ...] | None = None,
        chunk_tokens: int = 4096,
    ) -> None:
        if not hasattr(transformer, "blocks"):
            raise ValueError("Wan transformer has no blocks")
        self.groups: dict[str, AttentionBlockStats] = {}
        self._processors = []

        selected = [
            (index, block.attn1, block.attn1.processor)
            for index, block in enumerate(transformer.blocks)
            if block_indices is None or index in block_indices
        ]
        for index, _attention, processor in selected:
            if processor.qkv_observer is not None:
                raise RuntimeError(
                    f"blocks.{index}.attn1 already has a Q/K/V observer"
                )

        for index, attention, processor in selected:
            num_heads = attention.heads
            out_features = attention.to_q.out_features
            if out_features % num_heads:
                raise ValueError(
                    f"blocks.{index}.attn1 projection is not divisible by its heads"
                )
            accumulator_kwargs = {
                "num_heads": num_heads,
                "head_dim": out_features // num_heads,
                "device": device,
                "chunk_tokens": chunk_tokens,
            }
            group = AttentionBlockStats(
                query=ChannelStatsAccumulator(**accumulator_kwargs),
                key=ChannelStatsAccumulator(**accumulator_kwargs),
                value=ChannelStatsAccumulator(**accumulator_kwargs),
            )
            name = f"blocks.{index}.attn1"
            self.groups[name] = group

            def observe(
                _attention,
                query,
                key,
                value,
                *,
                acc=group,
                proc=processor,
            ):
                attention_kwargs = getattr(proc, "attention_kwargs", None)
                valid_kv_len = (
                    attention_kwargs.get("valid_kv_len")
                    if attention_kwargs is not None
                    else None
                )
                acc.add(
                    query,
                    key,
                    value,
                    valid_kv_len=valid_kv_len,
                )

            processor.qkv_observer = observe
            self._processors.append(processor)

    def merge_(self, other: "WanAttentionStatsObservers") -> None:
        if self.groups.keys() != other.groups.keys():
            raise ValueError("cannot merge different attention block sets")
        for name, group in self.groups.items():
            group.merge_(other.groups[name])

    def state_dict(self) -> dict[str, dict]:
        return {name: group.state_dict() for name, group in self.groups.items()}

    def remove(self) -> None:
        for processor in self._processors:
            processor.qkv_observer = None
        self._processors.clear()

    def __enter__(self) -> "WanAttentionStatsObservers":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
