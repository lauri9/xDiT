import math

import pytest
import torch
import torch.nn.functional as F

from xfuser.model_executor.quant.ptq.attn_mxfp4_ref import mxfp4_attention_ref
from xfuser.model_executor.quant.ptq.rotations import (
    HEAD_DIM, Rotation, apply_head_matrix, block_basis, dct_matrix, fold_input, fold_qk_signs,
    fold_value_output, fwht, hadamard, kernel_wht128, make_rotation, paley_hadamard,
    qk_prerotation, random_signs, sylvester_hadamard,
)


@pytest.mark.parametrize("order", [20, 32, 40, 108, 128])
def test_hadamard_constructions_are_hadamard(order):
    matrix = hadamard(order)
    assert set(matrix.unique().tolist()) == {-1.0, 1.0}
    torch.testing.assert_close(matrix @ matrix.T, order * torch.eye(order, dtype=torch.float64))


@pytest.mark.parametrize("order", [32, 128])
def test_paley_differs_from_sylvester_and_is_hadamard(order):
    matrix = paley_hadamard(order)
    torch.testing.assert_close(matrix @ matrix.T, order * torch.eye(order, dtype=torch.float64))
    assert not torch.equal(matrix, sylvester_hadamard(order))


@pytest.mark.parametrize("kind", ["wht", "wht_sequency", "paley", "hadamard", "dct", "haar"])
def test_block_bases_are_orthonormal(kind):
    basis = block_basis(kind, 32, seed=3)
    torch.testing.assert_close(basis @ basis.T, torch.eye(32, dtype=torch.float64), atol=1e-12, rtol=0)


def test_dct_matches_definition():
    x = torch.randn(3, 16, dtype=torch.float64)
    n = torch.arange(16, dtype=torch.float64)
    expected = torch.stack([(x * torch.cos(math.pi * (n + .5) * k / 16)).sum(-1) for k in range(16)], -1)
    expected = expected * math.sqrt(2 / 16)
    expected[:, 0] /= math.sqrt(2)
    torch.testing.assert_close(x @ dct_matrix(16), expected)


def test_fwht_matches_dense_sylvester():
    x = torch.randn(5, 256)
    dense = (sylvester_hadamard(128) / math.sqrt(128)).float()
    torch.testing.assert_close(fwht(x, 128), (x.reshape(5, 2, 128) @ dense).reshape(5, 256), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kernel_wht128(), dense)


ROTATIONS = [
    lambda: make_rotation(256, "identity"),
    lambda: make_rotation(256, "wht", block=32),
    lambda: make_rotation(256, "wht", block=128, sign_seed=4, sign_pairs=True),
    lambda: make_rotation(256, "wht_sequency", block=64),
    lambda: make_rotation(256, "paley", block=128, sign_seed=1),
    lambda: make_rotation(256, "dct", block=256),
    lambda: make_rotation(256, "haar", block=128, seed=7, perm=torch.randperm(256, generator=torch.Generator().manual_seed(2))),
    lambda: Rotation(80, "kron", inner=4, signs=random_signs(80, 9)),
    lambda: make_rotation(256, "dense", block=32, matrix=block_basis("haar", 32, seed=5)),
]


@pytest.mark.parametrize("build", ROTATIONS)
def test_rotation_apply_matches_dense_and_inverts(build):
    rotation = build()
    dense = rotation.dense()
    torch.testing.assert_close(dense @ dense.T, torch.eye(rotation.dim), atol=2e-6, rtol=0)
    x = torch.randn(7, rotation.dim)
    torch.testing.assert_close(rotation.apply(x), x @ dense, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(rotation.apply_transpose(rotation.apply(x)), x, atol=2e-5, rtol=2e-5)
    weight = torch.randn(11, rotation.dim)
    torch.testing.assert_close(rotation.apply(x) @ fold_input(weight, rotation).T, x @ weight.T, atol=2e-4, rtol=2e-4)
    clone = Rotation.from_state(rotation.state())
    assert clone.content_hash() == rotation.content_hash()
    torch.testing.assert_close(clone.apply(x), rotation.apply(x))


def test_kron_rotation_matches_explicit_kronecker():
    rotation = Rotation(80, "kron", inner=4)
    expected = torch.kron(hadamard(20) / math.sqrt(20), sylvester_hadamard(4) / 2).float()
    torch.testing.assert_close(rotation.dense(), expected, atol=1e-6, rtol=0)


def test_value_output_fold_preserves_attention_per_head_rotations():
    torch.manual_seed(0)
    heads, seq, width = 2, 9, 64
    values = torch.randn(1, seq, width, dtype=torch.float64).float()
    wv, bv = torch.randn(heads * HEAD_DIM, width), torch.randn(heads * HEAD_DIM)
    wo, bo = torch.randn(17, heads * HEAD_DIM), torch.randn(17)
    probabilities = torch.randn(1, heads, 5, seq).softmax(-1)

    def attention(wv, bv, wo):
        v = (values @ wv.T + bv).reshape(1, seq, heads, HEAD_DIM).transpose(1, 2)
        return (probabilities @ v).transpose(1, 2).reshape(1, 5, -1) @ wo.T + bo

    rotations = [make_rotation(HEAD_DIM, "haar", seed=1), make_rotation(HEAD_DIM, "wht", sign_seed=3)]
    torch.testing.assert_close(attention(*fold_value_output(wv, bv, wo, rotations)), attention(wv, bv, wo),
                               atol=2e-3, rtol=2e-4)


def _rope(values, cos, sin):
    x1, x2 = values.unflatten(-1, (-1, 2)).unbind(-1)
    out = torch.empty_like(values)
    out[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
    out[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
    return out


def test_pair_sign_fold_commutes_with_rms_norm_and_rope_exactly():
    heads, seq = 3, 6
    x = torch.randn(1, seq, 64)
    weight, bias = torch.randn(heads * HEAD_DIM, 64), torch.randn(heads * HEAD_DIM)
    gamma = torch.randn(heads * HEAD_DIM)
    angles = torch.randn(1, seq, 1, HEAD_DIM // 2).repeat_interleave(2, -1)
    cos, sin = angles.cos(), angles.sin()
    signs = random_signs(HEAD_DIM, 11, pairs=True)

    def pipeline(w, b):
        q = F.rms_norm(x @ w.T + b, (heads * HEAD_DIM,), gamma, 1e-6)
        return _rope(q.unflatten(2, (heads, HEAD_DIM)), cos, sin)

    folded = pipeline(*fold_qk_signs(weight, bias, signs))
    torch.testing.assert_close(folded, pipeline(weight, bias) * signs.float(), atol=0, rtol=0)
    with pytest.raises(ValueError, match="pairs"):
        fold_qk_signs(weight, bias, random_signs(HEAD_DIM, 1, pairs=False) * torch.tensor([1.0, -1.0]).repeat(64))


def _bshd(seed, seq=96, heads=2):
    generator = torch.Generator().manual_seed(seed)
    return [torch.randn(1, seq, heads, HEAD_DIM, generator=generator).bfloat16() for _ in range(3)]


def test_default_emulator_unchanged_and_explicit_wht_matches():
    q, k, v = _bshd(1)
    default = mxfp4_attention_ref(q, k, v)
    explicit = mxfp4_attention_ref(q, k, v, qk_rotation=kernel_wht128())
    torch.testing.assert_close(explicit.float(), default.float(), atol=2e-2, rtol=0)


@pytest.mark.parametrize("build", [
    lambda: make_rotation(HEAD_DIM, "identity"),
    lambda: make_rotation(HEAD_DIM, "wht", sign_seed=5, sign_pairs=True),
    lambda: make_rotation(HEAD_DIM, "wht", block=32),
    lambda: make_rotation(HEAD_DIM, "haar", seed=8),
])
def test_prerotation_reproduces_target_rotation_in_emulator(build):
    rotation = build()
    q, k, v = _bshd(2)
    pre = qk_prerotation(rotation)
    through_kernel_h = mxfp4_attention_ref(apply_head_matrix(q, pre).bfloat16(), apply_head_matrix(k, pre).bfloat16(), v)
    direct = mxfp4_attention_ref(q, k, v, qk_rotation=rotation.dense())
    reference = F.scaled_dot_product_attention(*(t.transpose(1, 2).float() for t in (q, k, v))).transpose(1, 2)
    error_pre = (through_kernel_h.float() - reference).square().sum() / reference.square().sum()
    error_direct = (direct.float() - reference).square().sum() / reference.square().sum()
    assert abs(float(error_pre) - float(error_direct)) <= 0.1 * float(error_direct)


def test_sign_prerotation_is_exact_in_bf16():
    rotation = make_rotation(HEAD_DIM, "wht", sign_seed=6, sign_pairs=True)
    q, _, _ = _bshd(3)
    torch.testing.assert_close(apply_head_matrix(q, qk_prerotation(rotation)).bfloat16(),
                               (q.float() * rotation.signs.float()).bfloat16(), atol=0, rtol=0)


def test_per_head_emulator_rotation_matches_shared():
    q, k, v = _bshd(4)
    shared = make_rotation(HEAD_DIM, "haar", seed=3).dense()
    torch.testing.assert_close(mxfp4_attention_ref(q, k, v, qk_rotation=shared),
                               mxfp4_attention_ref(q, k, v, qk_rotation=shared.expand(2, -1, -1)))
