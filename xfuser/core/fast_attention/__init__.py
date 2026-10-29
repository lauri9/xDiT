"""DiTFastAttn.

``fast_attn_state`` answers whether fast attention is on. That answer does not
need the attention implementation. Calibration lives in ``utils`` and imports
the layer package, which imports ``base_wrapper``, which asks for the flag. Loading
calibration from here at import time cycles. The implementation names below
resolve on first access.
"""

import importlib

from .fast_attn_state import (
    get_fast_attn_state,
    get_fast_attn_enable,
    get_fast_attn_step,
    get_fast_attn_calib,
    get_fast_attn_threshold,
    get_fast_attn_window_size,
    get_fast_attn_coco_path,
    get_fast_attn_use_cache,
    get_fast_attn_config_file,
    get_fast_attn_layer_name,
    initialize_fast_attn_state,
)

__all__ = [
    "get_fast_attn_state",
    "get_fast_attn_enable",
    "get_fast_attn_step",
    "get_fast_attn_calib",
    "get_fast_attn_threshold",
    "get_fast_attn_window_size",
    "get_fast_attn_coco_path",
    "get_fast_attn_use_cache",
    "get_fast_attn_config_file",
    "get_fast_attn_layer_name",
    "initialize_fast_attn_state",
    "xFuserFastAttention",
    "FastAttnMethod",
    "fast_attention_compression",
]

_LAZY = {
    "xFuserFastAttention": (".attn_layer", "xFuserFastAttention"),
    "FastAttnMethod": (".attn_layer", "FastAttnMethod"),
    "fast_attention_compression": (".utils", "fast_attention_compression"),
}


def __getattr__(name):
    exported = _LAZY.get(name)
    if exported is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = exported
    value = getattr(importlib.import_module(module_name, __package__), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
