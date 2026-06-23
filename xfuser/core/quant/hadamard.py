import torch

_VALID_BLOCK_R = frozenset({16, 32, 64, 128})
_GEMM_HADAMARD_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def build_hadamard_matrix(block_r: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Normalized Hadamard matrix (block_r x block_r, R @ R.T == I; block_r a power of two).

    Uses aiter's create_hadamard_matrix with a local Sylvester fallback.
    """
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                create_hadamard_matrix,
            )
        return (create_hadamard_matrix(block_r, dtype=dtype) / (block_r ** 0.5)).detach().cpu()
    except ImportError:
        assert block_r & (block_r - 1) == 0, "Hadamard block_r must be a power of 2"
        H = torch.ones((1, 1), dtype=torch.float32)
        while H.shape[0] < block_r:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        return (H / (block_r ** 0.5)).to(dtype)


def hadamard_rotate(x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Orthonormal Hadamard rotation along the last dimension in blocks of R.shape[-1]."""
    d = x.shape[-1]
    block_r = R.shape[-1]
    R = R.to(device=x.device, dtype=x.dtype)
    if block_r == d:
        return torch.matmul(x, R)
    return torch.matmul(x.unflatten(-1, (d // block_r, block_r)), R).flatten(-2)


def build_per_device_hadamard_matrix(
    block_r: int,
    dtype: torch.dtype = torch.bfloat16,
    *,
    allow_missing_aiter: bool = False,
) -> dict[torch.device, torch.Tensor | None]:
    """Replicate a Hadamard matrix on each available device."""
    hadamard_matrix: dict[torch.device, torch.Tensor | None] = {}
    try:
        _hadamard = build_hadamard_matrix(block_r, dtype=dtype)
    except Exception:
        if allow_missing_aiter:
            _hadamard = None
        else:
            raise
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            hadamard_matrix[device] = _hadamard.to(device) if _hadamard is not None else None
    else:
        device = torch.device("cpu")
        hadamard_matrix[device] = _hadamard.to(device) if _hadamard is not None else None
    return hadamard_matrix


def get_hadamard_matrix_for_device(block_r: int, device: torch.device) -> torch.Tensor:
    """Lazy per-(block_r, device) cache for GEMM Hadamard matrices."""
    if block_r not in _VALID_BLOCK_R:
        raise ValueError(f"Hadamard block_r must be one of {sorted(_VALID_BLOCK_R)}, got {block_r}")
    key = (block_r, device)
    if key not in _GEMM_HADAMARD_CACHE:
        R = build_hadamard_matrix(block_r)
        _GEMM_HADAMARD_CACHE[key] = R.to(device)
    return _GEMM_HADAMARD_CACHE[key]


def fold_weight_for_hadamard(weight: torch.Tensor, block_r: int) -> torch.Tensor:
    """Fold orthonormal rotation into weights: W' = W @ R along in_features."""
    if weight.shape[-1] % block_r != 0:
        raise ValueError(
            f"in_features ({weight.shape[-1]}) must be divisible by hadamard block_r ({block_r})"
        )
    R = get_hadamard_matrix_for_device(block_r, weight.device)
    return hadamard_rotate(weight, R).contiguous()
