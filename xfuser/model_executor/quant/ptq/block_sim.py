"""Single Wan 2.2 transformer block replay for MXFP4 rotation ablations.

Functional re-implementation of diffusers ``WanTransformerBlock`` (as run by
xDiT with ``xFuserWanAttnProcessor``) whose numerics can be switched per
site:

* every GEMM group is ``bf16``, ``emulated`` (exact ``mxfp4_ref`` RTN on
  activations and/or weights, FP32 accumulation) or ``kernel`` (AITER
  ``per_1x32`` + ``gemm_a4w4``, the production ``xFuserMXFP4Linear`` path);
* self-attention is MHA v4 ``bf16`` or ``mxfp4`` (cross-attention is always
  BF16, as in production), with an optional Q/K pre-rotation ``M = R·H``;
* :class:`RotationPlan` folds offline rotations into the weights and returns
  the online rotations that the GEMM inputs need.

``attention_impl="reference"`` replaces AITER with SDPA / the MXFP4 emulator
so the module is testable on CPU at small sequence lengths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .mxfp4_ref import dequantize_mxfp4, quantize_mxfp4
from .rotations import (
    HEAD_DIM, Rotation, apply_head_matrix, fold_input, fold_qk_signs, fold_value_output,
    kernel_wht128, qk_prerotation,
)

EPS = 1e-6
GEMM_GROUPS = {
    "self_qkv": ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    "self_out": ("attn1.to_out.0",),
    "cross_q": ("attn2.to_q",),
    "cross_kv": ("attn2.to_k", "attn2.to_v"),
    "cross_out": ("attn2.to_out.0",),
    "ffn_in": ("ffn.net.0.proj",),
    "ffn_out": ("ffn.net.2",),
}
LINEARS = tuple(name for names in GEMM_GROUPS.values() for name in names)
OTHER_PARAMS = ("attn1.norm_q.weight", "attn1.norm_k.weight", "attn2.norm_q.weight",
                "attn2.norm_k.weight", "norm2.weight", "norm2.bias", "scale_shift_table")


def parameter_names() -> list[str]:
    return [f"{name}.{kind}" for name in LINEARS for kind in ("weight", "bias")] + list(OTHER_PARAMS)


def load_block_params(component: str, index: int, device="cuda") -> dict[str, torch.Tensor]:
    """Read one block from the pinned checkpoint as BF16 (production dtype)."""
    from .calibrate import MODEL_ID, MODEL_REVISION
    from .quantize import _checkpoint_map, _read_weight

    weight_map = _checkpoint_map(MODEL_ID, MODEL_REVISION, component)
    return {name: _read_weight(weight_map, f"blocks.{index}.{name}", device).to(torch.bfloat16)
            for name in parameter_names()}


# ---------------------------------------------------------------------------
# Quantization numerics
# ---------------------------------------------------------------------------


def fake_quant_mxfp4(values: torch.Tensor, *, chunk_rows: int = 16384) -> torch.Tensor:
    """Exact ``dequant(quant(x))`` per 1x32 over the last dimension, FP32."""
    flat = values.reshape(-1, values.shape[-1])
    out = torch.empty(flat.shape, dtype=torch.float32, device=values.device)
    for start in range(0, flat.shape[0], chunk_rows):
        out[start:start + chunk_rows] = dequantize_mxfp4(*quantize_mxfp4(flat[start:start + chunk_rows]))
    return out.reshape(values.shape)


class KernelGemm:
    """The production ``xfuser::mxfp4_gemm`` path with a cached packed weight."""

    def __init__(self, weight: torch.Tensor):
        import aiter
        from aiter.ops.shuffle import shuffle_weight

        self._aiter = aiter
        self._quant = aiter.get_hip_quant(aiter.QuantType.per_1x32)
        packed, self.scale = self._quant(weight.to(torch.bfloat16).contiguous(), shuffle=True)
        self.packed = shuffle_weight(packed, layout=(16, 16))

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        flat = values.reshape(-1, values.shape[-1]).contiguous()
        quantized, scale = self._quant(flat, shuffle=True)
        out = self._aiter.gemm_a4w4(quantized, self.packed, scale, self.scale, bpreshuffle=True, bias=None)
        return out.view(*values.shape[:-1], -1)


@dataclass
class GemmSetting:
    mode: str = "bf16"  # bf16 | emulated | kernel
    quant_act: bool = True  # emulated only
    quant_weight: bool = True  # emulated only


@dataclass
class BlockConfig:
    gemms: dict[str, GemmSetting] = field(default_factory=dict)
    self_attention: str = "mxfp4"  # mxfp4 | bf16
    attention_impl: str = "aiter"  # aiter | reference

    @classmethod
    def uniform(cls, mode: str, self_attention: str, **kwargs) -> "BlockConfig":
        return cls({group: GemmSetting(mode) for group in GEMM_GROUPS}, self_attention, **kwargs)

    def gemm(self, group: str) -> GemmSetting:
        return self.gemms.get(group, GemmSetting())


# ---------------------------------------------------------------------------
# Rotation plans (offline folds + online rotations)
# ---------------------------------------------------------------------------


@dataclass
class RotationPlan:
    """Which rotation each site uses. ``None`` keeps production behaviour.

    ``qk`` is the 128-wide rotation the MXFP4 attention quantizer should see.
    A list gives one rotation per head.
    ``qk_mode="prerotate"`` feeds ``Q (R H)`` to the kernel; ``"fold_signs"``
    requires ``qk`` to be a WHT-128 with pair-equal signs and folds them into
    ``to_q``/``to_k`` (no runtime op).
    """

    qk: Rotation | list | torch.Tensor | None = None
    qk_mode: str = "prerotate"
    self_vo: Rotation | list | None = None
    cross_vo: Rotation | list | None = None
    online: dict[str, Rotation] = field(default_factory=dict)
    ffn_perm: torch.Tensor | None = None
    cross_kv_input: Rotation | None = None

    def describe(self) -> dict:
        def one(value):
            if value is None:
                return None
            if isinstance(value, Rotation):
                return value.describe()
            if isinstance(value, list):
                return [v.describe() for v in value]
            return {"matrix_shape": list(value.shape)}

        return {"qk": one(self.qk), "qk_mode": self.qk_mode, "self_vo": one(self.self_vo),
                "cross_vo": one(self.cross_vo), "online": {k: one(v) for k, v in self.online.items()},
                "ffn_perm": None if self.ffn_perm is None else True,
                "cross_kv_input": one(self.cross_kv_input)}


ONLINE_TARGETS = {
    "self_qkv": ("attn1.to_q", "attn1.to_k", "attn1.to_v"),
    "self_out": ("attn1.to_out.0",),
    "cross_q": ("attn2.to_q",),
    "cross_out": ("attn2.to_out.0",),
    "ffn_in": ("ffn.net.0.proj",),
    "ffn_out": ("ffn.net.2",),
}


@dataclass
class PreparedBlock:
    params: dict[str, torch.Tensor]
    online: dict[str, Rotation]
    qk_prerotation: torch.Tensor | None
    encoder_rotation: Rotation | None


def prepare_block(params: dict[str, torch.Tensor], plan: RotationPlan | None) -> PreparedBlock:
    """Fold the plan's offline rotations; weights stay FP32 until use."""
    plan = plan or RotationPlan()
    out = {name: value for name, value in params.items()}
    if plan.ffn_perm is not None:
        perm = plan.ffn_perm.to(out["ffn.net.2.weight"].device)
        out["ffn.net.0.proj.weight"] = out["ffn.net.0.proj.weight"].index_select(0, perm)
        out["ffn.net.0.proj.bias"] = out["ffn.net.0.proj.bias"].index_select(0, perm)
        out["ffn.net.2.weight"] = out["ffn.net.2.weight"].index_select(1, perm)
    for prefix, rotation in (("attn1", plan.self_vo), ("attn2", plan.cross_vo)):
        if rotation is None:
            continue
        wv, bv, wo = fold_value_output(out[f"{prefix}.to_v.weight"], out[f"{prefix}.to_v.bias"],
                                       out[f"{prefix}.to_out.0.weight"], rotation)
        out[f"{prefix}.to_v.weight"], out[f"{prefix}.to_v.bias"], out[f"{prefix}.to_out.0.weight"] = wv, bv, wo
    if plan.cross_kv_input is not None:
        for name in ("attn2.to_k", "attn2.to_v"):
            out[f"{name}.weight"] = fold_input(out[f"{name}.weight"], plan.cross_kv_input)
    for group, rotation in plan.online.items():
        if group not in ONLINE_TARGETS:
            raise ValueError(f"no online rotation site {group!r}")
        for name in ONLINE_TARGETS[group]:
            out[f"{name}.weight"] = fold_input(out[f"{name}.weight"], rotation)
    prerotation = None
    if plan.qk is not None:
        if plan.qk_mode == "fold_signs":
            rotations = plan.qk if isinstance(plan.qk, list) else [plan.qk]
            for rotation in rotations:
                if not isinstance(rotation, Rotation) or rotation.kind != "wht" or rotation.block != HEAD_DIM \
                        or rotation.perm is not None:
                    raise ValueError("fold_signs needs signed WHT-128 rotations")
            signs = torch.stack([torch.ones(HEAD_DIM, dtype=torch.float64) if r.signs is None else r.signs
                                 for r in rotations])
            signs = signs[0] if len(rotations) == 1 else signs
            for name in ("attn1.to_q", "attn1.to_k"):
                out[f"{name}.weight"], out[f"{name}.bias"] = fold_qk_signs(
                    out[f"{name}.weight"], out[f"{name}.bias"], signs)
        elif plan.qk_mode == "prerotate":
            prerotation = qk_prerotation(plan.qk, "cpu")
        else:
            raise ValueError(f"unknown qk_mode {plan.qk_mode!r}")
    return PreparedBlock(out, dict(plan.online), prerotation, plan.cross_kv_input)


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


def _rope(values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = values.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = cos[..., 0::2], sin[..., 1::2]
    out = torch.empty_like(values)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(values)


class WanBlockSim:
    def __init__(self, prepared: PreparedBlock, config: BlockConfig, *, heads: int = 40,
                 weight_overrides: dict[str, torch.Tensor] | None = None):
        self.config = config
        self.heads = heads
        self.online = prepared.online
        self.encoder_rotation = prepared.encoder_rotation
        self.p = {name: value for name, value in prepared.params.items()}
        self.prerotation = prepared.qk_prerotation
        self.overrides = weight_overrides or {}
        self._kernels: dict[str, KernelGemm] = {}
        self._fq_weights: dict[str, torch.Tensor] = {}
        self._norms = {}
        for group, linears in GEMM_GROUPS.items():
            setting = config.gemm(group)
            if setting.mode not in ("bf16", "emulated", "kernel"):
                raise ValueError(f"unknown GEMM mode {setting.mode!r}")
            for name in linears:
                weight = self.p[f"{name}.weight"].to(torch.bfloat16)
                if setting.mode == "kernel":
                    if name in self.overrides:
                        raise ValueError("weight overrides are emulated-mode only")
                    self._kernels[name] = KernelGemm(weight)
                elif setting.mode == "emulated" and setting.quant_weight:
                    self._fq_weights[name] = (self.overrides[name].float() if name in self.overrides
                                              else fake_quant_mxfp4(weight))

    # -- pieces ---------------------------------------------------------------------

    def linear(self, group: str, name: str, values: torch.Tensor) -> torch.Tensor:
        """Weights are used in the activation dtype (BF16 in production, so
        folded FP32 weights are stored BF16 exactly as an artifact would be)."""
        setting = self.config.gemm(group)
        bias = self.p[f"{name}.bias"].to(values.dtype)
        if setting.mode == "kernel":
            return self._kernels[name](values) + bias
        weight = self.p[f"{name}.weight"].to(values.dtype)
        if setting.mode == "bf16":
            return F.linear(values, weight, bias)
        x = fake_quant_mxfp4(values) if setting.quant_act else values.float()
        w = self._fq_weights[name] if setting.quant_weight else weight.float()
        return (x @ w.T).to(values.dtype) + bias

    def group_input(self, group: str, values: torch.Tensor) -> torch.Tensor:
        rotation = self.online.get(group)
        return values if rotation is None else rotation.apply(values).to(values.dtype)

    def _rms(self, name: str, values: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(values, (values.shape[-1],), self.p[name].to(values.dtype), EPS)

    def _qk_norm_rope(self, query, key, rotary):
        if query.is_cuda and self.config.attention_impl == "aiter":
            try:
                from xfuser.model_executor.layers.fused_qk_norm_rope_wan_flydsl import fused_qk_norm_rope
            except ImportError:
                fused_qk_norm_rope = None
            if fused_qk_norm_rope is not None:
                norms = []
                for name in ("attn1.norm_q.weight", "attn1.norm_k.weight"):
                    if name not in self._norms:
                        norm = torch.nn.RMSNorm(query.shape[-1], eps=EPS, elementwise_affine=True,
                                                device=query.device, dtype=torch.bfloat16)
                        norm.weight.data.copy_(self.p[name])
                        self._norms[name] = norm
                    norms.append(self._norms[name])
                return fused_qk_norm_rope(query, key, norms[0], norms[1], rotary[0], rotary[1], self.heads)
        query = self._rms("attn1.norm_q.weight", query).unflatten(2, (self.heads, -1))
        key = self._rms("attn1.norm_k.weight", key).unflatten(2, (self.heads, -1))
        return _rope(query, *rotary), _rope(key, *rotary)

    def attention(self, query, key, value, *, quantized: bool) -> torch.Tensor:
        """BSHD in, BSHD out."""
        if self.config.attention_impl == "aiter":
            from aiter.ops.mha_v4 import AttentionFormat, mha_v4

            fmt = AttentionFormat.MXFP4 if quantized else AttentionFormat.BF16
            return mha_v4(query.contiguous(), key.contiguous(), value.contiguous(), fmt, fmt, fmt)
        if quantized:
            from .attn_mxfp4_ref import mxfp4_attention_ref

            return mxfp4_attention_ref(query, key, value)
        out = F.scaled_dot_product_attention(query.transpose(1, 2).float(), key.transpose(1, 2).float(),
                                             value.transpose(1, 2).float())
        return out.transpose(1, 2).to(query.dtype)

    # -- forward ---------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, hidden, encoder, temb, rotary, *, taps: dict | None = None) -> torch.Tensor:
        """Run the block. ``taps``, when given, receives intermediate tensors;
        restrict them with ``taps["_want"] = {...}`` (default: all). Keys:
        ``self_qkv_in``, ``q``/``k``/``v`` (post-RoPE, before any pre-rotation),
        ``self_attn``, ``cross_q_in``, ``ffn_in_in``, ``ffn_out_in`` (GELU
        output). GEMM inputs are tapped before online rotations."""

        def tap(key, value):
            if taps is not None and (taps.get("_want") is None or key in taps["_want"]):
                taps[key] = value

        p = self.p
        dtype = hidden.dtype
        mods = (p["scale_shift_table"].float() + temb.float()).chunk(6, dim=1)
        shift, scale, gate, c_shift, c_scale, c_gate = mods
        if self.encoder_rotation is not None:
            encoder = self.encoder_rotation.apply(encoder).to(encoder.dtype)

        # 1. self-attention
        normed = (F.layer_norm(hidden.float(), (hidden.shape[-1],), eps=EPS) * (1 + scale) + shift).to(dtype)
        tap("self_qkv_in", normed)
        qkv_in = self.group_input("self_qkv", normed)
        query = self.linear("self_qkv", "attn1.to_q", qkv_in)
        key = self.linear("self_qkv", "attn1.to_k", qkv_in)
        value = self.linear("self_qkv", "attn1.to_v", qkv_in).unflatten(2, (self.heads, -1))
        query, key = self._qk_norm_rope(query, key, rotary)
        tap("q", query)
        tap("k", key)
        tap("v", value)
        quantized = self.config.self_attention == "mxfp4"
        if quantized and self.prerotation is not None:
            query = apply_head_matrix(query, self.prerotation).to(dtype)
            key = apply_head_matrix(key, self.prerotation).to(dtype)
        attended = self.attention(query, key, value, quantized=quantized).flatten(2, 3).to(dtype)
        tap("self_attn", attended)
        attn_out = self.linear("self_out", "attn1.to_out.0", self.group_input("self_out", attended))
        hidden = (hidden.float() + attn_out * gate).to(dtype)

        # 2. cross-attention (BF16 attention, as production)
        normed = F.layer_norm(hidden.float(), (hidden.shape[-1],), p["norm2.weight"].float(),
                              p["norm2.bias"].float(), EPS).to(dtype)
        tap("cross_q_in", normed)
        query = self.linear("cross_q", "attn2.to_q", self.group_input("cross_q", normed))
        key = self.linear("cross_kv", "attn2.to_k", encoder)
        value = self.linear("cross_kv", "attn2.to_v", encoder)
        query = self._rms("attn2.norm_q.weight", query).unflatten(2, (self.heads, -1))
        key = self._rms("attn2.norm_k.weight", key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        attended = self.attention(query, key, value, quantized=False).flatten(2, 3).to(dtype)
        hidden = hidden + self.linear("cross_out", "attn2.to_out.0", self.group_input("cross_out", attended))

        # 3. feed-forward
        normed = (F.layer_norm(hidden.float(), (hidden.shape[-1],), eps=EPS) * (1 + c_scale) + c_shift).to(dtype)
        tap("ffn_in_in", normed)
        inner = F.gelu(self.linear("ffn_in", "ffn.net.0.proj", self.group_input("ffn_in", normed)),
                       approximate="tanh")
        tap("ffn_out_in", inner)
        ff = self.linear("ffn_out", "ffn.net.2", self.group_input("ffn_out", inner))
        return (hidden.float() + ff.float() * c_gate).to(dtype)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class UpdateError:
    """Pooled error energies of the block update ``out - in``.

    ``add_pair`` adds one conditional/unconditional pair and accumulates
    per-branch and CFG-combined (``g Δc - (g-1) Δu``) error and signal.
    """

    def __init__(self):
        self.sums = {key: 0.0 for key in ("cfg_error2", "cfg_signal2", "branch_error2", "branch_signal2",
                                          "output_error2", "output_signal2")}
        self.pairs = 0

    def add_pair(self, inputs, references, candidates, *, guidance: float) -> dict:
        deltas = [r.float() - x.float() for r, x in zip(references, inputs)]
        errors = [c.float() - r.float() for c, r in zip(candidates, references)]
        cfg_ref = guidance * deltas[0] - (guidance - 1) * deltas[1]
        cfg_err = guidance * errors[0] - (guidance - 1) * errors[1]
        record = {
            "cfg_error2": float(cfg_err.square().sum()), "cfg_signal2": float(cfg_ref.square().sum()),
            "branch_error2": float(sum(e.square().sum() for e in errors)),
            "branch_signal2": float(sum(d.square().sum() for d in deltas)),
            "output_error2": float(sum(e.square().sum() for e in errors)),
            "output_signal2": float(sum(r.float().square().sum() for r in references)),
        }
        for key, value in record.items():
            self.sums[key] += value
        self.pairs += 1
        return {**record, "cfg_update_rel_mse": record["cfg_error2"] / record["cfg_signal2"],
                "branch_update_rel_mse": record["branch_error2"] / record["branch_signal2"]}

    def report(self) -> dict:
        s = self.sums
        return {"pairs": self.pairs, **s,
                "cfg_update_rel_mse": s["cfg_error2"] / s["cfg_signal2"] if s["cfg_signal2"] else None,
                "branch_update_rel_mse": s["branch_error2"] / s["branch_signal2"] if s["branch_signal2"] else None,
                "output_rel_mse": s["output_error2"] / s["output_signal2"] if s["output_signal2"] else None}


def reference_kernel_wht(device="cpu") -> torch.Tensor:
    """Exported for drivers: the production Q/K rotation (``R = H``)."""
    return kernel_wht128(device)
