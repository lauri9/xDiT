import json
from types import SimpleNamespace

import pytest
import torch

from xfuser.model_executor.quant.ptq import evaluate
from xfuser.model_executor.quant.ptq.evaluate import (
    AttentionDecompositionHooks,
    ErrorSums,
    _callback,
    _eval_args,
    _sample_rows,
    _velocity_metrics,
    aggregate_attention_reports,
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


def test_attention_arms_hold_gemms_in_bf16_and_change_only_self_attention():
    reference = _eval_args("attn_bf16", None)
    candidate = _eval_args("attn_mxfp4", None)

    assert reference.gemm_quantization_spec.is_pure("none")
    assert candidate.gemm_quantization_spec.is_pure("none")
    assert reference.attention_backend == "AITER_BF16"
    assert candidate.attention_backend == "AITER_MXFP4"
    assert reference.cross_attention_backend == "AITER_BF16"
    assert candidate.cross_attention_backend == "AITER_BF16"


def test_velocity_metrics_uses_wan_conditional_then_unconditional_call_order():
    reference = [torch.tensor([[2.0]]), torch.tensor([[1.0]])]
    candidate = [torch.tensor([[2.0]]), torch.tensor([[0.0]])]

    report = _velocity_metrics(reference, candidate, guidance=4.0)

    assert report["cfg_combined"]["relative_mse"] == pytest.approx(0.36)


def test_attention_aggregate_applies_small_error_and_p_dominance_gates(tmp_path):
    def prompt(error):
        return {
            "rung3": {
                "velocity": {
                    expert: {"cfg_combined": {"relative_mse": error}}
                    for expert in ("transformer", "transformer_2")
                }
            },
            "rung4": {
                "latent_drift": [{"relative_mse": error}],
                "frame_ssim": 0.9,
                "frame_psnr_db": [20.0, 22.0],
            },
        }

    report_path = tmp_path / "attention.json"
    rtn_path = tmp_path / "rtn.json"
    decomposition_path = tmp_path / "decomposition.json"
    output_path = tmp_path / "summary.json"
    report_path.write_text(
        '{"arms":{"attn_mxfp4":{"prompts":['
        + json.dumps(prompt(0.01))
        + "]}}}"
    )
    rtn_path.write_text(
        '{"arms":{"rtn":{"prompts":['
        + json.dumps(prompt(0.2))
        + "]}}}"
    )
    decomposition_path.write_text(
        json.dumps(
            [
                {
                    "errors": {
                        "q_only": {"relative_mse": 0.1},
                        "k_only": {"relative_mse": 0.1},
                        "v_only": {"relative_mse": 0.1},
                        "p_only": {"relative_mse": 0.5},
                        "all": {"relative_mse": 0.8},
                    }
                }
            ]
        )
    )

    summary = aggregate_attention_reports(
        [report_path], [rtn_path], decomposition_path, output_path
    )

    assert summary["rung3"]["mean_error_ratio_to_rtn_gemm"] == pytest.approx(0.05)
    assert summary["decomposition"]["p_dominates"] is True
    assert summary["gate"]["decision"] == "stop"


def test_attention_decomposition_hooks_select_fixed_blocks_and_calls(monkeypatch):
    def transformer():
        return SimpleNamespace(
            blocks=[
                SimpleNamespace(
                    attn1=SimpleNamespace(
                        processor=SimpleNamespace(qkv_observer=None)
                    )
                )
                for _ in range(40)
            ]
        )

    pipe = SimpleNamespace(
        transformer=transformer(),
        transformer_2=transformer(),
    )

    def decompose(query, _key, _value, query_indices):
        reference = query.index_select(1, query_indices)
        return {
            "reference": reference,
            "q_only": reference + 1,
            "k_only": reference + 1,
            "v_only": reference + 1,
            "p_only": reference + 1,
            "all": reference + 1,
        }

    monkeypatch.setattr(evaluate, "mxfp4_attention_decomposition", decompose)
    hooks = AttentionDecompositionHooks(pipe, query_rows=2)
    observed = pipe.transformer.blocks[0].attn1.processor.qkv_observer
    tensor = torch.zeros((1, 5, 2, 128))

    observed(None, tensor, tensor, tensor)
    rows = hooks.remove()

    assert len(rows) == 1
    assert rows[0]["expert"] == "transformer"
    assert rows[0]["block"] == 0
    assert rows[0]["call"] == 0
    assert rows[0]["query_indices"] == [0, 4]
    assert all(
        block.attn1.processor.qkv_observer is None
        for component in (pipe.transformer, pipe.transformer_2)
        for block in (component.blocks[0], component.blocks[20], component.blocks[39])
    )
