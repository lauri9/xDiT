"""Wan linear-input Hessian observers for GPTQ calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class WanHessianGroup:
    name: str
    source: str
    linears: tuple[str, ...]


def wan_block_hessian_groups(block_index: int) -> tuple[WanHessianGroup, ...]:
    prefix = f"blocks.{block_index}"
    return (
        WanHessianGroup(
            f"{prefix}.self_qkv",
            f"{prefix}.attn1.to_q",
            tuple(f"{prefix}.attn1.to_{name}" for name in ("q", "k", "v")),
        ),
        WanHessianGroup(
            f"{prefix}.self_out",
            f"{prefix}.attn1.to_out.0",
            (f"{prefix}.attn1.to_out.0",),
        ),
        WanHessianGroup(
            f"{prefix}.cross_q",
            f"{prefix}.attn2.to_q",
            (f"{prefix}.attn2.to_q",),
        ),
        WanHessianGroup(
            f"{prefix}.cross_kv",
            f"{prefix}.attn2.to_k",
            (f"{prefix}.attn2.to_k", f"{prefix}.attn2.to_v"),
        ),
        WanHessianGroup(
            f"{prefix}.cross_out",
            f"{prefix}.attn2.to_out.0",
            (f"{prefix}.attn2.to_out.0",),
        ),
        WanHessianGroup(
            f"{prefix}.ffn_in",
            f"{prefix}.ffn.net.0.proj",
            (f"{prefix}.ffn.net.0.proj",),
        ),
        WanHessianGroup(
            f"{prefix}.ffn_out",
            f"{prefix}.ffn.net.2",
            (f"{prefix}.ffn.net.2",),
        ),
    )


class HessianAccumulator:
    """Accumulate X^T X and per-channel activation statistics in fp32."""

    def __init__(
        self,
        in_features: int,
        *,
        device: torch.device | str,
        chunk_tokens: int = 16384,
        token_sample_rate: float = 1.0,
        tokens_per_forward: int | None = None,
        seed: int = 0,
    ) -> None:
        if not 0 < token_sample_rate <= 1:
            raise ValueError("token_sample_rate must be in (0, 1]")
        if tokens_per_forward is not None and tokens_per_forward <= 0:
            raise ValueError("tokens_per_forward must be positive")
        self.in_features = in_features
        self.device = torch.device(device)
        self.chunk_tokens = chunk_tokens
        self.token_sample_rate = token_sample_rate
        self.tokens_per_forward = tokens_per_forward
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.hessian = torch.zeros(
            (in_features, in_features),
            dtype=torch.float32,
            device=self.device,
        )
        self.absmax = torch.zeros(in_features, dtype=torch.float32, device=self.device)
        self.sum_squares = torch.zeros(in_features, dtype=torch.float64, device=self.device)
        self.token_count = 0

    def _sample(self, values: torch.Tensor) -> torch.Tensor:
        count = values.shape[0]
        wanted = count
        if self.token_sample_rate < 1:
            wanted = min(wanted, max(1, round(count * self.token_sample_rate)))
        if self.tokens_per_forward is not None:
            wanted = min(wanted, self.tokens_per_forward)
        if wanted == count:
            return values
        indices = torch.randperm(count, generator=self.generator, device=values.device)[:wanted]
        return values[indices]

    @torch.no_grad()
    def add(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                f"observer expected K={self.in_features}, got {inputs.shape[-1]}"
            )
        values = inputs.reshape(-1, self.in_features)
        if mask is not None:
            flat_mask = mask.reshape(-1).to(device=values.device, dtype=torch.bool)
            if flat_mask.numel() != values.shape[0]:
                raise ValueError("observer mask does not match the input token dimensions")
            values = values[flat_mask]
        if not values.numel():
            return
        values = self._sample(values)
        for chunk in values.split(self.chunk_tokens):
            chunk_f32 = chunk.to(device=self.device, dtype=torch.float32)
            self.hessian.addmm_(chunk_f32.T, chunk_f32)
            torch.maximum(
                self.absmax,
                chunk_f32.abs().amax(dim=0),
                out=self.absmax,
            )
            self.sum_squares.add_((chunk_f32.to(torch.float64) ** 2).sum(dim=0))
            self.token_count += chunk_f32.shape[0]

    @property
    def rms(self) -> torch.Tensor:
        if not self.token_count:
            return torch.zeros_like(self.absmax)
        return (self.sum_squares / self.token_count).sqrt().to(torch.float32)

    def merge_(self, other: "HessianAccumulator") -> None:
        if self.in_features != other.in_features:
            raise ValueError("cannot merge Hessians with different K dimensions")
        self.hessian.add_(other.hessian.to(self.device))
        torch.maximum(self.absmax, other.absmax.to(self.device), out=self.absmax)
        self.sum_squares.add_(other.sum_squares.to(self.device))
        self.token_count += other.token_count


class WanHessianObservers:
    """Install one hook for each of the seven unique linear inputs per block."""

    def __init__(
        self,
        transformer: nn.Module,
        *,
        device: torch.device | str,
        mask_provider: Callable[[str, torch.Tensor], torch.Tensor | None] | None = None,
        block_indices: tuple[int, ...] | None = None,
        **accumulator_kwargs,
    ) -> None:
        if not hasattr(transformer, "blocks"):
            raise ValueError("Wan transformer has no blocks")
        self.transformer = transformer
        self.mask_provider = mask_provider
        self.groups: dict[str, HessianAccumulator] = {}
        self.linear_to_group: dict[str, str] = {}
        self._handles = []

        for block_index, block in enumerate(transformer.blocks):
            if block_indices is not None and block_index not in block_indices:
                continue
            for attention_name in ("attn1", "attn2"):
                attention = getattr(block, attention_name)
                if getattr(attention, "fused_projections", False):
                    raise ValueError(f"blocks.{block_index}.{attention_name} has fused projections")
            for group in wan_block_hessian_groups(block_index):
                source = transformer.get_submodule(group.source)
                if type(source) is not nn.Linear:
                    raise TypeError(
                        f"{group.source} is {type(source).__name__}, expected an unfused nn.Linear"
                    )
                accumulator = HessianAccumulator(
                    source.in_features,
                    device=device,
                    **accumulator_kwargs,
                )
                self.groups[group.name] = accumulator
                for linear_name in group.linears:
                    linear = transformer.get_submodule(linear_name)
                    if type(linear) is not nn.Linear:
                        raise TypeError(
                            f"{linear_name} is {type(linear).__name__}, expected nn.Linear"
                        )
                    if linear.in_features != source.in_features:
                        raise ValueError(f"shared Hessian inputs disagree for {group.name}")
                    self.linear_to_group[linear_name] = group.name

                def observe(_module, args, *, name=group.name, acc=accumulator):
                    values = args[0]
                    mask = self.mask_provider(name, values) if self.mask_provider else None
                    acc.add(values, mask)

                self._handles.append(source.register_forward_pre_hook(observe))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "WanHessianObservers":
        return self

    def __exit__(self, *_exc) -> None:
        self.remove()
