import pytest
import torch

from xfuser.model_executor.quant.ptq.evaluate import (
    ErrorSums,
    _callback,
    _eval_args,
    _sample_rows,
    _velocity_metrics,
)


def test_error_sums_reports_relative_error_sqnr_and_cosine():
    sums = ErrorSums()
    sums.update(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 1.0]))

    report = sums.report()

    assert report["relative_mse"] == pytest.approx(0.2)
    assert report["sqnr_db"] == pytest.approx(6.9897, rel=1e-4)
    assert report["cosine_similarity"] == pytest.approx(3 / (5 * 2) ** 0.5)


def test_callback_records_candidate_and_forces_reference_next_latent():
    recorded = []
    forced = [torch.full((2,), 7.0)]
    callback = _callback(recorded, forced)
    candidate = torch.tensor([1.0, 2.0])

    result = callback(None, 0, None, {"latents": candidate})

    torch.testing.assert_close(recorded[0], candidate)
    torch.testing.assert_close(result["latents"], forced[0])


def test_sample_rows_is_deterministic_and_bounded():
    values = torch.arange(40).reshape(10, 4)

    sampled = _sample_rows(values, count=3)

    assert sampled.shape == (3, 4)
    torch.testing.assert_close(sampled, values[[0, 4, 9]])


def test_rtn_uses_pure_fp4_for_both_wan_experts():
    args = _eval_args("rtn", None)

    assert args.gemm_quantization_spec.is_pure("fp4")
    assert args.use_fp4_gemms


def test_velocity_metrics_uses_wan_conditional_then_unconditional_call_order():
    reference = [torch.tensor([[2.0]]), torch.tensor([[1.0]])]
    candidate = [torch.tensor([[2.0]]), torch.tensor([[0.0]])]

    report = _velocity_metrics(reference, candidate, guidance=4.0)

    assert report["cfg_combined"]["relative_mse"] == pytest.approx(0.36)
