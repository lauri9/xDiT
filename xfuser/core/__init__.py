"""Core pieces. Long-context attention imports yunchang, which imports AITER on ROCm.

Importing ``xfuser.core.distributed`` loads this package first. Keep that path free of
yunchang by resolving these names on first access.
"""

__all__ = [
    "CacheManager",
    "xFuserLongContextAttention",
    "xFuserSanaLinearLongContextAttention",
    "AttnType",
    "gpu_timer_decorator",
]

_LAZY = {
    "CacheManager": ("xfuser.core.cache_manager", "CacheManager"),
    "xFuserLongContextAttention": ("xfuser.core.long_ctx_attention", "xFuserLongContextAttention"),
    "xFuserSanaLinearLongContextAttention": (
        "xfuser.core.long_ctx_attention",
        "xFuserSanaLinearLongContextAttention",
    ),
    "AttnType": ("xfuser.core.long_ctx_attention", "AttnType"),
    "gpu_timer_decorator": ("xfuser.core.utils", "gpu_timer_decorator"),
}


def __getattr__(name):
    exported = _LAZY.get(name)
    if exported is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attr = exported
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
