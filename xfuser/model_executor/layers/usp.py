# This file implements USP with torch version >= '2.5.0'
import os
import torch
import torch.distributed as dist
import functools

import torch.distributed._functional_collectives as ft_c

from torch.distributed.tensor.experimental._attention import _templated_ring_attention
import xfuser.envs as envs

if torch.cuda.is_available() or envs._is_npu():
    from yunchang.globals import PROCESS_GROUP
else:
    PROCESS_GROUP = None

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_rank,
    get_runtime_state,
)

from packaging.version import parse
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.attention_backend import (
    ATTENTION_FUNCTION_REGISTRY,
    AttentionBackendType,
)
from xfuser.core.sparge_attention import head_balance

# Sparge backends whose kernel cost can be load-balanced across Ulysses ranks.
# These all build a block mask via _build_sparge_block_mask and write the
# per-head cost into the head-balance "cost sink". Non-sparge backends are
# excluded so head balancing is a clean no-op for them.
_HEAD_BALANCE_BACKENDS = frozenset({
    AttentionBackendType.AITER_SPARGE,
    AttentionBackendType.AITER_SPARGE_ASM,
    AttentionBackendType.AITER_SPARGE_ASM_FP8,
    AttentionBackendType.AITER_SPARGE_ASM_V2,
    AttentionBackendType.AITER_SPARGE_V2,
})

_FP8_LOG_SCALES = bool(os.environ.get("XFUSER_FP8_LOG_SCALES"))
_FP8_NCCL_NEEDS_VIEW = parse(torch.__version__).release < parse("2.11.0").release
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz)


def ring_attn(attention_function, query, key, value, dropout_p=0.0, is_causal=False, joint_attn_kwargs=None, attention_kwargs=None):
    kwargs = {
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "joint_attn_kwargs": joint_attn_kwargs,
        "attention_kwargs": attention_kwargs,
    }
    if parse(torch.__version__).release >= parse("2.6.0").release:
        from torch.distributed.tensor.experimental._attention import _cp_options
        _cp_options.enable_load_balance = False
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            1,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    else:
        out, *_ = _templated_ring_attention(
            PROCESS_GROUP.RING_PG,
            attention_function,
            query,
            key,
            value,
            **kwargs,
        )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x_dtype = x.dtype
    x = x.flatten()
    # NCCL does not support FP8 collectives before PyTorch 2.11, view as uint8 (same width) for the transfer.
    if _FP8_NCCL_NEEDS_VIEW and x_dtype in _FP8_DTYPES:
        x = x.view(torch.uint8)
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.view(x_dtype).reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _per_tensor_quant(x: torch.Tensor, scale_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize x to FP8 using a fixed pre-allocated scale tensor. Returns (x_fp8, descale)."""
    import aiter
    fp8_dtype = aiter.dtypes.fp8
    return aiter.per_tensor_quant(x, scale=scale_t, quant_dtype=fp8_dtype, dtypeMax=torch.finfo(fp8_dtype).max)



def _fp8_comms_input_all_to_all(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple:
    """Quantize Q/K/V to FP8 using shared scales and run interleaved input all-to-alls.

    Scales start at 1.0 and are updated once per generation via all_reduce after step 1.
    Returns (query, key, value, attn_kwargs_update, (q_scale, k_scale, v_scale), qkv_amaxes).
    """
    fp8_comms = get_runtime_state().fp8_comms
    q_scale, k_scale, v_scale = fp8_comms.q_scale, fp8_comms.k_scale, fp8_comms.v_scale

    # quantize with shared synced scale so all ranks agree on encoding, then all-to-all
    q_fp8, q_descale = _per_tensor_quant(query, q_scale)
    query = _ft_c_input_all_to_all(q_fp8)
    k_fp8, k_descale = _per_tensor_quant(key, k_scale)
    key = _ft_c_input_all_to_all(k_fp8)
    v_fp8, v_descale = _per_tensor_quant(value, v_scale)
    value = _ft_c_input_all_to_all(v_fp8)

    qkv_amaxes = (
        (q_descale.item(), k_descale.item(), v_descale.item())
        if _FP8_LOG_SCALES else None
    )

    attn_kwargs_update = {
        "pre_quantized": True,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
    }
    return query, key, value, attn_kwargs_update, (q_scale, k_scale, v_scale), qkv_amaxes


def _fp8_comms_output_all_to_all(out: torch.Tensor, v_scale_t: torch.Tensor) -> torch.Tensor:
    """Quantize attention output to FP8, run output all-to-all, dequantize back."""
    restore_dtype = out.dtype if out.dtype not in _FP8_DTYPES else torch.bfloat16
    if out.dtype not in _FP8_DTYPES:
        out_fp8, out_descale = _per_tensor_quant(out, v_scale_t)
    else:
        out_fp8, out_descale = out, v_scale_t
    return (_ft_c_output_all_to_all(out_fp8).float() * out_descale).to(restore_dtype)


def _combined_qkv_all_to_all(q, k, v):
    """Concatenate query, key, value tensors and perform a single all-to-all communication."""
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return q, k, v

    assert q.ndim == 4, f"q must have 4 dimensions, got {q.ndim}"
    b, h, s, d = q.shape
    assert h % world_size == 0, f"h must be divisible by world_size, got {h} and {world_size}"

    # [3, b, h, s, d]
    qkv = torch.stack([q, k, v], dim=0)
    # [3, b, P, h/P, s, d]
    qkv = qkv.view(3, b, world_size, h // world_size, s, d)
    # [P, 3, b, h/P, s, d]
    qkv = qkv.permute(2, 0, 1, 3, 4, 5).contiguous()

    qkv = _sdpa_all_to_all_single(qkv)

    # [3, b, h/P, P*s, d]  — reshape directly avoids the intermediate
    # contiguous copy that the separate permute+view required.
    qkv = qkv.permute(1, 2, 3, 0, 4, 5).reshape(3, b, h // world_size, -1, d)

    q, k, v = torch.unbind(qkv, dim=0)
    return q, k, v


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x


def _preprocess_joint_tensors(joint_key, joint_value):
    """
    Preprocess the joint key and value tensors for Ulysses parallelism.
    """
    ulysses_world_size = get_ulysses_parallel_world_size()
    ulysses_rank = get_ulysses_parallel_rank()
    attn_heads_per_ulysses_rank = (
        joint_key.shape[1] // ulysses_world_size
    )
    joint_key = joint_key.transpose(1,2)
    joint_value = joint_value.transpose(1,2)
    joint_key = joint_key[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :, ].transpose(1,2)
    joint_value = joint_value[
        ...,
        attn_heads_per_ulysses_rank
        * ulysses_rank : attn_heads_per_ulysses_rank
        * (ulysses_rank + 1),
        :,
    ].transpose(1,2)
    return joint_key, joint_value

def _concat_joint_tensor(tensor, joint_tensor, joint_strategy, dim):
    """
    Concatenate the joint tensor to the main tensor based on the joint strategy.
    """
    if joint_strategy == "rear":
        tensor = torch.cat([tensor, joint_tensor], dim=dim)
    elif joint_strategy == "front":
        tensor = torch.cat([joint_tensor, tensor], dim=dim)
    else:
        raise ValueError(f"Invalid joint_strategy: {joint_strategy}")
    return tensor

def _update_and_get_kv_cache(key, value, attn_layer):
    """
    Update and get the key and value cache for pipeline parallelism.
    """
    key, value = get_cache_manager().update_and_get_kv_cache(
        new_kv=[key.transpose(1, 2), value.transpose(1, 2)],
        layer=attn_layer,
        slice_dim=1,
        layer_type="attn",
    )
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()
    return key, value

def _get_attention_function(backend=None):
    """
    Get the attention function based on the runtime state or from a given explicit backend.
    """
    if backend is not None:
        attention_backend = backend
    else:
        attention_backend = get_runtime_state().attention_backend
    func = ATTENTION_FUNCTION_REGISTRY.get(attention_backend, None)
    if func is None:
        raise NotImplementedError(f"Attention backend {attention_backend} not registered.")
    return concat_joint_tensors_decorator(func)

def concat_joint_tensors_decorator(func):
    """
    Decorator to handle joint tensor concatenation
    This is needed for ring attention with 'rear' joint_strategy, as it
    needs to concat the joint tensors before calling the attention function
    but only on the last step.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        query, key, value = args[0:3]
        is_causal = kwargs.get("is_causal")
        dropout_p = kwargs.get("dropout_p")
        joint_attn_kwargs = kwargs.get("joint_attn_kwargs", None)
        attention_kwargs = kwargs.get("attention_kwargs", None)

        if joint_attn_kwargs is not None:
            joint_strategy = joint_attn_kwargs.get("joint_strategy", None)
            joint_key = joint_attn_kwargs.get("joint_key", None)
            joint_value = joint_attn_kwargs.get("joint_value", None)
            step = joint_attn_kwargs.get("step", 0)
            total_steps = joint_attn_kwargs.get("total_steps", 1)
            if (joint_strategy == "front" and step == 0) or (joint_strategy == "rear" and step == total_steps - 1):
                key = _concat_joint_tensor(key, joint_key, joint_strategy, dim=2)
                value = _concat_joint_tensor(value, joint_value, joint_strategy, dim=2)
            joint_attn_kwargs["step"] = step + 1 # In place increment step

        return func(query, key, value, dropout_p=dropout_p, is_causal=is_causal, attention_kwargs=attention_kwargs)
    return wrapper


def USP(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        joint_query: torch.Tensor | None = None,
        joint_key: torch.Tensor | None = None,
        joint_value: torch.Tensor | None = None,
        joint_strategy: str | None = None,
        attn_layer=None,
        combine_qkv_a2a: bool | None = None,
        use_fp8_comms: bool = False,
        backend=None,
        attention_kwargs: dict | None = None,
        head_balance_layer=None,
    ):
    """
    Unified Sequence Parallelism (USP) attention call, supporting combinations of Ulysses and
    Ring attention. Also supports joint tensors and key-value caching for pipeline parallelism.
    Explicit backend can be provided to specify the attention backend to use.

    ``head_balance_layer`` (optional): a stable per-layer handle (e.g. the
    attention module). When provided and --use_spargeattn_head_balance is set, the
    Ulysses head dimension is permuted so each rank gets a cost-balanced subset
    of heads (block-sparse load balancing); the permutation is inverted on the
    output. No-op for non-sparse backends (no cost is published) and for ring/
    joint paths.
    """
    if combine_qkv_a2a is None:
        combine_qkv_a2a = False

    attention_function = _get_attention_function(backend=backend)

    # ---- Ulysses block-sparse head load balancing ----
    # Permute the heads before the input all-to-all so each rank receives a
    # cost-balanced subset, then invert on the output. The permutation for this
    # step is read from a per-layer buffer (filled from the previous step's
    # per-head cost).
    hb_uly = get_ulysses_parallel_world_size()
    hb_perm = getattr(head_balance_layer, "head_perm", None)
    hb_backend = backend if backend is not None else get_runtime_state().attention_backend
    hb_active = (
        head_balance.ENABLED
        and hb_perm is not None
        and joint_strategy is None
        and hb_uly > 1
        and get_ring_parallel_world_size() == 1
        and query.shape[1] % hb_uly == 0
        and hb_backend in _HEAD_BALANCE_BACKENDS
    )
    hb_inv = None
    hb_cost_sink = None
    if hb_active:
        hb_perm = hb_perm.clone()  # snapshot the permutation applied this step
        hb_inv = torch.argsort(hb_perm)
        query = query.index_select(1, hb_perm)
        key = key.index_select(1, hb_perm)
        value = value.index_select(1, hb_perm)
        # Scratch tensor for the sparse backend to write this rank's per-head
        # cost into (shape = heads-per-rank). Passed via a shallow-copied
        # attention_kwargs. Written in-place by the backend; read
        # back below..
        hb_cost_sink = query.new_zeros(query.shape[1] // hb_uly, dtype=torch.float32)
    else:
        hb_perm = None

    joint_attn_kwargs = None
    if joint_strategy:
        query = _concat_joint_tensor(query, joint_query, joint_strategy, dim=2)
        joint_key, joint_value = _preprocess_joint_tensors(joint_key, joint_value)
        joint_attn_kwargs = {
            "joint_value": joint_value,
            "joint_key": joint_key,
            "joint_strategy": joint_strategy,
            "step": 0,
            "total_steps": get_ring_parallel_world_size(),

        }

    qkv_scales = None
    qkv_amaxes = None
    if get_ulysses_parallel_world_size() > 1:
        if use_fp8_comms:
            fp8_comms_backend = backend if backend is not None else get_runtime_state().attention_backend
            if fp8_comms_backend == AttentionBackendType.AITER_FP8:
                from xfuser.core.distributed.attention_backend import (
                    FP8_HADAMARD_MATRIX,
                    _fp8_hadamard_rotate,
                )
                R = FP8_HADAMARD_MATRIX[query.device]
                query = _fp8_hadamard_rotate(query, R).contiguous()
                key = _fp8_hadamard_rotate(key, R).contiguous()
            query, key, value, attn_kwargs_update, qkv_scales, qkv_amaxes = _fp8_comms_input_all_to_all(query, key, value)
            attention_kwargs = (attention_kwargs or {}) | attn_kwargs_update
        elif combine_qkv_a2a and query.shape == key.shape == value.shape:
            query, key, value = _combined_qkv_all_to_all(query, key, value)
        else:
            query = _ft_c_input_all_to_all(query)
            key = _ft_c_input_all_to_all(key)
            value = _ft_c_input_all_to_all(value)

    if hb_active:
        hb_attention_kwargs = {
            **(attention_kwargs or {}),
            head_balance.COST_SINK_KEY: hb_cost_sink,
        }
    else:
        hb_attention_kwargs = attention_kwargs

    if attn_layer:
        key, value = _update_and_get_kv_cache(key, value, attn_layer)

    if get_sequence_parallel_world_size() == 1: # No SP
        out, _ = attention_function(query,
                                    key,
                                    value,
                                    dropout_p=dropout_p,
                                    is_causal=is_causal,
                                    joint_attn_kwargs=joint_attn_kwargs,
                                    attention_kwargs=attention_kwargs)

    elif get_ulysses_parallel_world_size() == 1: # Ring only
        out = ring_attn(attention_function,
                        query,
                        key,
                        value,
                        dropout_p=dropout_p,
                        is_causal=is_causal,
                        joint_attn_kwargs=joint_attn_kwargs,
                        attention_kwargs=attention_kwargs)

    else:
        if get_ring_parallel_world_size() == 1: # Ulysses only
            out, _ = attention_function(query,
                                        key,
                                        value,
                                        dropout_p=dropout_p,
                                        is_causal=is_causal,
                                        joint_attn_kwargs=joint_attn_kwargs,
                                        attention_kwargs=hb_attention_kwargs)
        else: # USP
            out = ring_attn(attention_function,
                            query,
                            key,
                            value,
                            dropout_p=dropout_p,
                            is_causal=is_causal,
                            joint_attn_kwargs=joint_attn_kwargs,
                            attention_kwargs=attention_kwargs)
        if use_fp8_comms:
            if _FP8_LOG_SCALES and qkv_amaxes is not None:
                out_amax = out.abs().amax().item()
                rank = dist.get_rank()
                q_amax, k_amax, v_amax = qkv_amaxes
                print(f"[fp8_scales rank{rank}] q_amax={q_amax:.4f} k_amax={k_amax:.4f} v_amax={v_amax:.4f} out_amax={out_amax:.4f}")
            _, _, v_scale_t = qkv_scales
            out = _fp8_comms_output_all_to_all(out, v_scale_t)
        else:
            out = _ft_c_output_all_to_all(out)
        if hb_active:
            # Restore the original (global) head order on the output.
            out = out.index_select(1, hb_inv)
            # hb_cost_sink now holds this rank's per-head cost (written in place
            # by the sparse backend). Exchange across the Ulysses group via a
            # functional collective, map back to global
            # head order, and store next step's balanced permutation into the
            # per-layer buffer.
            full = _maybe_wait(
                ft_c.all_gather_tensor(hb_cost_sink, 0, PROCESS_GROUP.ULYSSES_PG)
            )
            glob = head_balance.scatter_to_global(full, hb_perm)
            head_balance_layer.head_perm.copy_(
                head_balance.compute_perm(glob, hb_uly)
            )

    return out


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        use_fp8_comms: bool = False,  # accepted for call-site uniformity with USP(), never applied
        backend=None,
        attention_kwargs=None,
        head_balance_layer=None,
    ):
    """
    Runs attention call without any parallelism.
    This can be used when the logic necessitates no Ulysses or Ring parallelism in any case.
    Explicit backend can be provided to specify the attention backend to use.

    ``head_balance_layer`` is accepted for call-site signature parity with
    ``USP`` but ignored here: with no Ulysses parallelism there is no head
    sharding to balance.
    """
    attention_function = _get_attention_function(backend=backend)
    out, _ = attention_function(
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal,
        attention_kwargs=attention_kwargs,
    )
    return out

