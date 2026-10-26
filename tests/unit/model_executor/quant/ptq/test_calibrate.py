from types import SimpleNamespace

import pytest

from xfuser.model_executor.quant.ptq.calibrate import (
    CalibrationSettings,
    assert_no_prompt_overlap,
    build_eager_bf16_args,
    preflight_eager_bf16,
    prompt_indices,
    prompt_seed,
    read_prompts,
)


def test_prompt_seeds_do_not_depend_on_rank_sharding(tmp_path):
    settings = CalibrationSettings(prompts_path=tmp_path / "prompts.txt")
    all_indices = set()

    for rank in range(8):
        shard = prompt_indices(50, rank, 8)
        assert all_indices.isdisjoint(shard)
        all_indices.update(shard)
        assert [prompt_seed(settings, index) for index in shard] == [
            42 + index for index in shard
        ]

    assert all_indices == set(range(50))


def test_prompt_file_and_overlap_validation(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first\nsecond\n", encoding="utf-8")

    assert read_prompts(path) == ["first", "second"]
    assert_no_prompt_overlap(["first"], ["held out"])
    with pytest.raises(ValueError, match="overlap"):
        assert_no_prompt_overlap(["first"], ["first", "held out"])


def test_eager_bf16_args_disable_quantization_compile_cache_and_parallelism(tmp_path):
    settings = CalibrationSettings(prompts_path=tmp_path / "prompts.txt")

    args = build_eager_bf16_args(settings, world_size=8)

    assert args.model_revision == settings.model_revision
    assert args.data_parallel_degree == 8
    assert args.ulysses_degree == args.ring_degree == args.tensor_parallel_degree == 1
    assert args.pipefusion_parallel_degree == args.fully_shard_degree == 1
    assert not args.use_torch_compile
    assert not args.use_fp4_gemms
    assert not args.use_fp8_gemms
    assert not args.use_fp8_comms
    assert not args.use_hybrid_attn_schedule
    assert not args.use_hybrid_gemm_schedule
    assert args.cache_method is None


def test_preflight_rejects_an_active_optimization(tmp_path):
    settings = CalibrationSettings(prompts_path=tmp_path / "prompts.txt")
    config = build_eager_bf16_args(settings, world_size=1)
    config.use_torch_compile = True
    model = SimpleNamespace(
        config=config,
        pipe=SimpleNamespace(transformer=object(), transformer_2=object()),
    )

    with pytest.raises(RuntimeError, match="unexpectedly active"):
        preflight_eager_bf16(model, settings)


def test_calibration_requires_exact_revision_and_cfg(tmp_path):
    with pytest.raises(ValueError, match="exact 40-character"):
        CalibrationSettings(
            prompts_path=tmp_path / "prompts.txt",
            model_revision="main",
        )
    with pytest.raises(ValueError, match="both CFG"):
        CalibrationSettings(
            prompts_path=tmp_path / "prompts.txt",
            guidance_scale=1.0,
        )
