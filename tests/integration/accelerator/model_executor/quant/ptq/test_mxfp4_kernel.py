import pytest
import torch

from xfuser.model_executor.layers.mxfp4_linear import xFuserMXFP4Linear
from xfuser.model_executor.quant.ptq.mxfp4_ref import quantize_mxfp4
from xfuser.model_executor.quant.ptq.prequantized_load import (
    _shuffle_scale,
    _shuffle_weight,
)


pytestmark = [pytest.mark.accelerator, pytest.mark.rocm]


def test_reference_is_bit_exact_with_aiter_per_1x32():
    aiter = pytest.importorskip("aiter")
    generator = torch.Generator(device="cuda").manual_seed(42)
    values = torch.randn(
        (257, 64),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    values[0].zero_()
    values[1, :8] = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0],
        dtype=torch.bfloat16,
        device="cuda",
    )

    packed_ref, scale_ref = quantize_mxfp4(values)
    quantize = aiter.get_hip_quant(aiter.QuantType.per_1x32)
    packed_aiter, scale_aiter = quantize(values, shuffle=False)

    torch.testing.assert_close(
        packed_ref,
        packed_aiter.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        scale_ref,
        scale_aiter.view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_unshuffled_rtn_artifact_matches_on_the_fly_gemm():
    generator = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn(
        (256, 256),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    inputs = torch.randn(
        (128, 256),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )

    baseline = xFuserMXFP4Linear(
        256,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        baseline.weight.copy_(weight)
    baseline._quantize_weights()

    packed, scale = quantize_mxfp4(weight)
    artifact = xFuserMXFP4Linear(
        256,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    artifact.install_packed_placeholders(
        weight_shape=(256, 128),
        scale_shape=(256, 8),
    )
    with torch.no_grad():
        artifact.weight_shuffle.copy_(_shuffle_weight(packed))
        artifact.weight_scale.copy_(_shuffle_scale(scale))

    torch.testing.assert_close(
        artifact.weight_shuffle.view(torch.uint8),
        baseline.weight_shuffle.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        artifact.weight_scale.view(torch.uint8),
        baseline.weight_scale.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(artifact(inputs), baseline(inputs), rtol=0, atol=0)
