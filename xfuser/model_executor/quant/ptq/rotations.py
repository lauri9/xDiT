"""Orthogonal rotations for the Wan 2.2 MXFP4 rotation ablations.

Convention: activations are rows and a rotation ``R`` maps ``x -> x @ R``.
A linear ``y = x @ W.T`` is preserved by folding ``W' = W @ R`` into the
consumer, because ``(x R)(W R)^T = x W^T`` for orthogonal ``R``.

A :class:`Rotation` is ``R = P · D · B`` applied left to right: an optional
channel permutation ``P`` (``x[..., perm]``), optional ±1 signs ``D`` and a
block-diagonal orthonormal ``B`` (or a full-width Kronecker Hadamard).
Every construction here is deterministic from its description, so the same
rotation can be rebuilt by the emulator, the block simulator and the xDiT
runtime from a manifest.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field

import torch

HEAD_DIM = 128


# ---------------------------------------------------------------------------
# Hadamard and other orthonormal bases (unnormalized ±1 Hadamards are cached
# in float64 on CPU; callers normalize).
# ---------------------------------------------------------------------------


def _is_prime(value: int) -> bool:
    return value > 1 and all(value % d for d in range(2, int(value**0.5) + 1))


@functools.lru_cache(maxsize=None)
def sylvester_hadamard(order: int) -> torch.Tensor:
    """Unnormalized ±1 Sylvester Hadamard (natural order; symmetric)."""
    if order < 1 or order & (order - 1):
        raise ValueError("Sylvester Hadamard order must be a power of two")
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < order:
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    return matrix


@functools.lru_cache(maxsize=None)
def paley_hadamard(order: int) -> torch.Tensor:
    """Unnormalized Paley-I Hadamard of order q+1 for prime q ≡ 3 (mod 4)."""
    q = order - 1
    if not _is_prime(q) or q % 4 != 3:
        raise ValueError(f"Paley-I needs order-1 prime and ≡ 3 mod 4, got {order}")
    residues = {(x * x) % q for x in range(1, q)}
    chi = torch.tensor([0] + [1 if d in residues else -1 for d in range(1, q)], dtype=torch.float64)
    index = torch.arange(q)
    jacobsthal = chi[(index.view(1, -1) - index.view(-1, 1)) % q]
    skew = torch.zeros(order, order, dtype=torch.float64)
    skew[0, 1:] = 1
    skew[1:, 0] = -1
    skew[1:, 1:] = jacobsthal
    return torch.eye(order, dtype=torch.float64) + skew


@functools.lru_cache(maxsize=None)
def hadamard(order: int) -> torch.Tensor:
    """Unnormalized ±1 Hadamard: Sylvester for powers of two, otherwise
    ``H_2^k ⊗ Paley(m)`` with the smallest Paley-I base that fits."""
    if order & (order - 1) == 0:
        return sylvester_hadamard(order)
    power = 1
    while order % (2 * power) == 0:
        try:
            base = paley_hadamard(order // power)
            return torch.kron(sylvester_hadamard(power), base)
        except ValueError:
            power *= 2
    try:
        return torch.kron(sylvester_hadamard(power), paley_hadamard(order // power))
    except ValueError as error:
        raise ValueError(f"no supported Hadamard construction of order {order}") from error


def sequency_permutation(order: int) -> torch.Tensor:
    """Column order that sorts Sylvester Walsh functions by sign changes."""
    matrix = sylvester_hadamard(order)
    changes = (matrix[:, 1:] != matrix[:, :-1]).sum(1)
    return torch.argsort(changes, stable=True)


def dct_matrix(order: int) -> torch.Tensor:
    """Orthonormal DCT-II as a right-multiplied rotation: ``x @ C = DCT(x)``."""
    i = torch.arange(order, dtype=torch.float64).view(-1, 1)
    k = torch.arange(order, dtype=torch.float64).view(1, -1)
    matrix = torch.cos(math.pi * (i + 0.5) * k / order) * math.sqrt(2 / order)
    matrix[:, 0] /= math.sqrt(2)
    return matrix


def haar_matrix(order: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(order, order, generator=generator, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def random_signs(dim: int, seed: int, *, pairs: bool = False) -> torch.Tensor:
    """±1 vector; with ``pairs`` both channels of each adjacent pair agree,
    which keeps the signs commuting with Wan's adjacent-pair RoPE."""
    generator = torch.Generator().manual_seed(seed)
    count = dim // 2 if pairs else dim
    signs = torch.randint(0, 2, (count,), generator=generator).double() * 2 - 1
    return signs.repeat_interleave(2) if pairs else signs


def block_basis(kind: str, block: int, seed: int | None = None) -> torch.Tensor | None:
    """Orthonormal ``block × block`` basis in float64; ``None`` is identity."""
    if kind == "identity":
        return None
    if kind == "wht":
        return sylvester_hadamard(block) / math.sqrt(block)
    if kind == "wht_sequency":
        return sylvester_hadamard(block)[:, sequency_permutation(block)] / math.sqrt(block)
    if kind == "paley":
        return paley_hadamard(block) / math.sqrt(block)
    if kind == "hadamard":
        return hadamard(block) / math.sqrt(block)
    if kind == "dct":
        return dct_matrix(block)
    if kind == "haar":
        if seed is None:
            raise ValueError("haar basis requires a seed")
        return haar_matrix(block, seed)
    raise ValueError(f"unknown rotation kind {kind!r}")


def fwht(values: torch.Tensor, size: int) -> torch.Tensor:
    """Normalized natural-order FWHT within last-dimension blocks (FP32)."""
    if size & (size - 1) or values.shape[-1] % size:
        raise ValueError("FWHT size must be a power of two dividing the last dimension")
    shape = values.shape
    work = values.float().reshape(-1, size)
    stride = 1
    while stride < size:
        pairs = work.reshape(-1, size // (2 * stride), 2, stride)
        first, second = pairs[:, :, 0], pairs[:, :, 1]
        work = torch.stack((first + second, first - second), dim=2).reshape(-1, size)
        stride *= 2
    return (work / math.sqrt(size)).reshape(shape)


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


@dataclass
class Rotation:
    """``R = P · D · B`` over the last dimension of width ``dim``.

    ``kind`` names the block basis (see :func:`block_basis`), or ``kron`` for a
    full-width ``hadamard(dim // inner) ⊗ sylvester(inner)``, or ``dense`` for
    an explicit ``block × block`` matrix in ``matrix`` (e.g. learned).
    """

    dim: int
    kind: str = "identity"
    block: int = 0
    seed: int | None = None
    signs: torch.Tensor | None = None
    perm: torch.Tensor | None = None
    matrix: torch.Tensor | None = None
    inner: int = HEAD_DIM
    label: str = ""
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.kind == "kron":
            self.block = self.dim
        elif self.kind == "identity" and not self.block:
            self.block = self.dim
        if self.block <= 0 or self.dim % self.block:
            raise ValueError("rotation block must divide the dimension")
        if self.kind == "dense":
            if self.matrix is None or tuple(self.matrix.shape) != (self.block, self.block):
                raise ValueError("dense rotation needs a block × block matrix")
            self.matrix = self.matrix.detach().double().cpu()
        if self.signs is not None:
            self.signs = self.signs.detach().double().cpu()
            if self.signs.shape != (self.dim,) or not bool((self.signs.abs() == 1).all()):
                raise ValueError("signs must be a ±1 vector of the rotation width")
        if self.perm is not None:
            self.perm = self.perm.detach().long().cpu()
            if not torch.equal(torch.sort(self.perm).values, torch.arange(self.dim)):
                raise ValueError("perm must be a permutation of the rotation width")

    # -- construction helpers -------------------------------------------------

    @classmethod
    def identity(cls, dim: int) -> "Rotation":
        return cls(dim, "identity", dim, label="identity")

    @property
    def is_identity(self) -> bool:
        return self.kind == "identity" and self.signs is None and self.perm is None

    def _basis(self, device, dtype=torch.float32) -> torch.Tensor | None:
        key = ("basis", str(device), dtype)
        if key not in self._cache:
            if self.kind == "dense":
                basis = self.matrix
            elif self.kind == "kron":
                basis = None
            else:
                basis = block_basis(self.kind, self.block, self.seed)
            self._cache[key] = None if basis is None else basis.to(device=device, dtype=dtype)
        return self._cache[key]

    def _kron_outer(self, device) -> torch.Tensor:
        key = ("outer", str(device))
        if key not in self._cache:
            outer = self.dim // self.inner
            self._cache[key] = (hadamard(outer) / math.sqrt(outer)).to(device=device, dtype=torch.float32)
        return self._cache[key]

    # -- application ------------------------------------------------------------

    def _block_apply(self, values: torch.Tensor, *, transpose: bool) -> torch.Tensor:
        if self.kind == "identity":
            return values
        if self.kind == "kron":
            outer = self._kron_outer(values.device)
            work = values.reshape(*values.shape[:-1], self.dim // self.inner, self.inner)
            work = fwht(work, self.inner)  # symmetric, self-transpose
            work = torch.einsum("...kj,kl->...lj", work, outer.T if transpose else outer)
            return work.reshape(values.shape)
        if self.kind == "wht":
            return fwht(values, self.block)
        basis = self._basis(values.device)
        if transpose:
            basis = basis.T
        work = values.reshape(*values.shape[:-1], self.dim // self.block, self.block)
        return (work @ basis).reshape(values.shape)

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values @ R`` in FP32 over the last dimension."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"rotation width {self.dim} does not match {values.shape[-1]}")
        work = values.float()
        if self.perm is not None:
            work = work.index_select(-1, self.perm.to(work.device))
        if self.signs is not None:
            work = work * self.signs.to(device=work.device, dtype=work.dtype)
        return self._block_apply(work, transpose=False)

    def apply_transpose(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values @ R.T`` (the inverse)."""
        if values.shape[-1] != self.dim:
            raise ValueError(f"rotation width {self.dim} does not match {values.shape[-1]}")
        work = self._block_apply(values.float(), transpose=True)
        if self.signs is not None:
            work = work * self.signs.to(device=work.device, dtype=work.dtype)
        if self.perm is not None:
            inverse = torch.empty_like(self.perm)
            inverse[self.perm] = torch.arange(self.dim)
            work = work.index_select(-1, inverse.to(work.device))
        return work

    def dense(self, device="cpu") -> torch.Tensor:
        """Explicit ``dim × dim`` FP32 matrix; only for small widths."""
        if self.dim > 8192:
            raise ValueError("refusing to materialize a rotation wider than 8192")
        return self.apply(torch.eye(self.dim, device=device))

    # -- provenance ---------------------------------------------------------------

    def describe(self) -> dict:
        def digest(tensor):
            if tensor is None:
                return None
            return hashlib.sha256(tensor.detach().double().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]

        return {"label": self.label, "dim": self.dim, "kind": self.kind, "block": self.block,
                "seed": self.seed, "inner": self.inner if self.kind == "kron" else None,
                "signs_sha": digest(self.signs), "perm_sha": digest(self.perm),
                "matrix_sha": digest(self.matrix)}

    def state(self) -> dict:
        return {"dim": self.dim, "kind": self.kind, "block": self.block, "seed": self.seed,
                "signs": self.signs, "perm": self.perm, "matrix": self.matrix,
                "inner": self.inner, "label": self.label}

    @classmethod
    def from_state(cls, state: dict) -> "Rotation":
        return cls(**state)

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.describe(), sort_keys=True).encode()).hexdigest()[:16]


def make_rotation(dim: int, kind: str, *, block: int | None = None, sign_seed: int | None = None,
                  sign_pairs: bool = False, seed: int | None = None, perm=None, matrix=None,
                  label: str | None = None) -> Rotation:
    """Convenience constructor used by the experiment drivers."""
    signs = None if sign_seed is None else random_signs(dim, sign_seed, pairs=sign_pairs)
    block = dim if block is None else block
    name = label or "-".join(str(part) for part in (
        kind, block if kind not in ("identity", "kron") else None,
        None if sign_seed is None else f"{'pair' if sign_pairs else ''}sign{sign_seed}",
        None if seed is None else f"seed{seed}", "perm" if perm is not None else None) if part is not None)
    return Rotation(dim, kind, block, seed=seed, signs=signs, perm=perm, matrix=matrix, label=name)


# ---------------------------------------------------------------------------
# Folds (function-preserving weight transforms; all return FP32)
# ---------------------------------------------------------------------------


def fold_input(weight: torch.Tensor, rotation: Rotation) -> torch.Tensor:
    """``W' = W R`` for a consumer whose input is rotated online by ``R``."""
    return rotation.apply(weight)


def fold_value_output(value_weight, value_bias, output_weight, rotation: Rotation | list[Rotation]):
    """Per-head V→O fold: ``Wv' = Rᵀ Wv``, ``bv' = bv R``, ``Wo' = Wo R``.

    ``rotation`` is one 128-wide rotation shared by all heads, or one per head.
    """
    heads = value_weight.shape[0] // HEAD_DIM
    rotations = rotation if isinstance(rotation, list) else [rotation] * heads
    if len(rotations) != heads or any(r.dim != HEAD_DIM for r in rotations):
        raise ValueError("V→O fold needs 128-wide rotations, one or one per head")
    if output_weight.shape[1] != value_weight.shape[0]:
        raise ValueError("V and O widths differ")

    def per_head(values):  # values [..., heads*128] -> rotate each head chunk
        chunks = values.float().unflatten(-1, (heads, HEAD_DIM))
        return torch.stack([r.apply(chunks[..., h, :]) for h, r in enumerate(rotations)], -2).flatten(-2)

    new_value = per_head(value_weight.float().T).T.contiguous()
    new_bias = None if value_bias is None else per_head(value_bias)
    return new_value, new_bias, per_head(output_weight).contiguous()


def fold_qk_signs(weight, bias, signs: torch.Tensor):
    """Fold per-head channel signs into ``to_q``/``to_k`` output rows.

    Exact through Wan's ``rms_norm_across_heads`` (sign-invariant RMS, diagonal
    affine) and adjacent-pair RoPE when the signs agree within each pair.
    """
    signs = signs.double().cpu()
    if signs.shape != (HEAD_DIM,) or not bool((signs[0::2] == signs[1::2]).all()):
        raise ValueError("Q/K sign fold needs 128 signs that agree within RoPE pairs")
    full = signs.repeat(weight.shape[0] // HEAD_DIM).to(weight.device, torch.float32)
    return (weight.float() * full.view(-1, 1)).contiguous(), None if bias is None else bias.float() * full


def kernel_wht128(device="cpu") -> torch.Tensor:
    """The normalized Sylvester WHT that AITER MHA v4 fuses into Q/K quantization."""
    return (sylvester_hadamard(HEAD_DIM) / math.sqrt(HEAD_DIM)).to(device=device, dtype=torch.float32)


def qk_prerotation(rotation: Rotation | torch.Tensor, device="cpu") -> torch.Tensor:
    """Matrix ``M = R H`` such that feeding ``Q M`` to MHA v4 quantizes ``Q R``.

    ``H`` is symmetric and involutory, so the kernel's fused ``· H`` gives
    ``Q R H H = Q R``. Accepts a 128-wide :class:`Rotation` or a ``[128,128]``
    / ``[heads,128,128]`` matrix ``R``.
    """
    matrix = rotation.dense(device) if isinstance(rotation, Rotation) else rotation.to(device).float()
    if matrix.shape[-2:] != (HEAD_DIM, HEAD_DIM):
        raise ValueError("Q/K rotations are 128 × 128 per head")
    return matrix @ kernel_wht128(device)


def apply_head_matrix(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``values[..., h, :] @ matrix[h]`` for BSHD values; ``matrix`` may be shared."""
    if matrix.ndim == 2:
        return values.float() @ matrix.to(values.device)
    return torch.einsum("bshd,hde->bshe", values.float(), matrix.to(values.device))
