# Accelerator test suite: wall-time findings

This note records the problems found while investigating the wall time of the
accelerator tests, especially the multi-GPU tests. Each problem has a separate
section so that a different session can address it alone.

## Branch status

Branch `perf/persistent-rank-pool` on `lauri9/xDiT`, based on `c6940b3`:

- `6d77d2a`: copy of the rename from
  [xdit-project/xDiT#1010](https://github.com/xdit-project/xDiT/pull/1010).
  Remove this commit when #1010 merges.
- `c5c4dfa`: the rank pool prototype (problem 2) and the `fresh_ranks`
  marker on the SDMA modules (problem 3). Work in progress; do not merge.
- The commit that adds this note.

The next work starts with problems 4 and 5, and then a pool run of the
multi-GPU selector. See "Rank pool prototype" for the commands.

## Environment of the measurements

- Commit: `c6940b3` on `main`, plus the rename from
  [xdit-project/xDiT#1010](https://github.com/xdit-project/xDiT/pull/1010).
- Image: `amdsiloai/pytorch-xdit-staging:b69c7a4-temp` (local build before
  MIOpen tuning), torch `2.13.0+git3369fa2`, HIP `7.18.26443`.
- Node: 8 AMD GPUs, scheduled with `brr run --exclusive`. Other users' jobs
  ran on the node between my runs, but not during my exclusive allocations.
- Multi-GPU selector: `tests/integration/accelerator -m "multi_gpu and not nvidia"`
  collects 159 tests.
- Raw logs and JUnit XML: `/tmp/flux-sp-baseline/` on node
  `cv350-odcdh2-c09-1`. They are not in the repository, and a reboot of that
  node deletes them. This note has the numbers that matter.

## Summary

| # | Problem | Effect | Caused by the rank pool? |
| --- | --- | --- | --- |
| 1 | MOSS `reduce-overhead` tests hang in `destroy_process_group()` | Up to 30 min for each of 4 tests | No |
| 2 | Each spawned test pays about 10 s of process overhead | Most of the multi-GPU test time | No |
| 3 | SDMA transport caches peer groups for the process lifetime | Hang when ranks are reused | Exposed by it |
| 4 | `test_qwen_image21_ulysses` replaces `usp.get_runtime_state` and does not restore it | Next test fails when ranks are reused | Exposed by it |
| 5 | `test_wan_sequence_parallel_pad` patches global attention specs on rank 0 only | Ranks diverge and hang when reused | Exposed by it |
| 6 | `test_vdn_ulysses_parity` spawns 2 ranks but has no `multi_gpu` marker | Test never runs | No |
| 7 | Two test modules had the same basename | Combined selector fails at collection | No (fixed by #1010) |
| 8 | Reused ranks keep 2 spinning threads after a test | CPU load between tests | Exposed by it |
| 9 | A hung rank fails a test only at its timeout | 600 s or 1800 s for each hang | No |

## 1. MOSS `reduce-overhead` tests hang in teardown

Tests: `tests/integration/accelerator/model_executor/test_moss_image_sequence_parallel.py::test_sequence_parallel_matches_one_gpu[*-reduce-overhead]`
(4 tests on ROCm: `AITER` and `SDPA_EFFICIENT`, each with `u4` and `u4-cfg2`).

Symptom:

- All ranks finish the compiled forward passes. Rank 0 writes `output.pt` a
  few seconds after the ranks start.
- Then all ranks block. Their main threads wait in a futex and use no CPU.
- The test fails only when the spawn timeout (`timeout=1800`) expires.
- I saw this with the current spawn code (stopped after 28 min) and with the
  rank pool (stopped after 5 min). Thus the pool does not cause it.
- The comment in the test says that `reduce-overhead` compilation "alone takes
  ~15 min". In my runs, the compilation took less than 1 min. The long time
  is probably this hang.

Stack of each rank (from `faulthandler`):

```text
torch/distributed/distributed_c10d.py:2407 in destroy_process_group   # pg_to_shutdown.shutdown()
xfuser/core/distributed/parallel_state.py:624 in destroy_distributed_environment
tests/.../test_moss_image_sequence_parallel.py:82 in _worker            # finally block
```

Line 2407 calls `shutdown()` on each process group. The comment there says
that `ncclCommAbort()` was a collective call in some NCCL versions.

Suggested next steps:

1. Reproduce one test alone:
   `python -m pytest "tests/integration/accelerator/model_executor/test_moss_image_sequence_parallel.py::test_sequence_parallel_matches_one_gpu[AITER-u4-reduce-overhead]"`
   on 4 GPUs.
2. Find out if collectives captured in CUDA graphs keep RCCL work in flight at
   teardown. Try `torch.cuda.synchronize()` and `torch._dynamo.reset()` (which
   frees the cudagraph trees) before `destroy_model_parallel()`.
3. Find out if this also hangs a long-lived runner that uses
   `reduce-overhead` and then destroys the distributed environment.

Effect: 4 tests at up to 30 min each, so up to 2 h for each multi-GPU run.
This is probably the largest single item in the multi-GPU wall time.

## 2. Per-test process overhead in `_spawn.py`

Each multi-GPU test spawns new interpreters
(`tests/integration/accelerator/_spawn.py`). On the tiny FLUX test
(`test_flux_sequence_parallel.py`), a test takes about 11.7 s, but the test
body takes about 1 s. Phases of one test (rank 0, warm caches):

| Phase | Time |
| --- | --- |
| Spawn and `import torch` | 1.1 s |
| Import the test module (xFuser, diffusers) | 4.0 to 4.3 s |
| First body in a new process (CUDA context, RCCL, first kernels) | 3.0 to 3.7 s |
| Same body again in the same process | 0.9 to 1.7 s |
| Process exit after the result | 2.1 s |
| Wait for an empty result queue (`result_queue.get(timeout=1)`) | 1.0 s |

Warm caches save only about 1 s for each test.

### Rank pool prototype

The local branch `perf/persistent-rank-pool` contains a prototype. It does
not change the default behavior.

- `XFUSER_TEST_RANK_POOL=1` makes `spawn_accelerator_ranks` use one pool of
  rank processes for the session, with one rank for each visible GPU. A test
  with world size N runs on ranks 0 to N-1.
- After each test, the pool destroys the parallel state, sets
  `runtime_state._RUNTIME` to `None`, calls `torch._dynamo.reset()`, and
  restores `os.environ` and the working directory.
- If a rank fails or dies, the test fails within about 5 s and the next test
  gets a new pool.
- A rank that misses its deadline gets `SIGUSR1` first and prints the stacks of
  all its threads (`faulthandler`).
- The marker `fresh_ranks` (registered in `pytest.ini`) runs a test in new
  processes even when the pool is on.

Commands (in the test container, through brr):

```bash
# Current spawn code
brr run --exclusive --gpus 8 -- python -m pytest tests/integration/accelerator \
  -m "multi_gpu and not nvidia" -k "not reduce-overhead" --durations=0 --junitxml=base.xml
# Rank pool
XFUSER_TEST_RANK_POOL=1 brr run --exclusive --gpus 8 -- python -m pytest tests/integration/accelerator \
  -m "multi_gpu and not nvidia" -k "not reduce-overhead" --durations=0 --junitxml=pool.xml
```

Files changed: `pytest.ini`, `tests/integration/accelerator/_spawn.py`,
`tests/integration/accelerator/conftest.py`, and the three SDMA test modules
(problem 3).

Results, multi-GPU selector without the 4 MOSS `reduce-overhead` tests:

| Run | Sum of test times |
| --- | --- |
| Current spawn code, 155 tests | 2177 s |
| Rank pool, 155 tests | 840 s (2.6x) |
| Current spawn code, without the 2 tests that fail with the pool | 2130 s |
| Rank pool, without those 2 tests | 589 s (3.6x) |

Most model and sequence-parallel files are 3x to 15x faster. For example:

| File | Current | Pool |
| --- | --- | --- |
| `test_sequence_parallel_padding.py` (38 tests) | 434 s | 73 s |
| `test_flux_sequence_parallel.py` (19 tests) | 219 s | 34 s |
| `test_prism_parallel.py` (4 tests) | 116 s | 11 s |

The three `fresh_ranks` SDMA files stay at about 216 s.

Before you enable the pool, fix problems 3, 4 and 5, and then run the
multi-GPU selector with the pool again until it passes.

Smaller change without the pool: read exactly `world_size` results from
`result_queue` instead of waiting 1 s for an empty queue. This saves about
1 s for each spawned test.

## 3. SDMA transport keeps stale peer groups

Location: `xfuser/core/distributed/sdma/transport.py`.

- `_groups` (line 80) maps a tuple of global ranks to a `PeerGroup`.
  `_by_process_group` (line 81) maps a process group to a `PeerGroup`.
  `_slot` and `_call` (lines 82 and 83) are counters for the process.
- `group_for()` (line 378) returns the cached `PeerGroup` for a tuple of ranks.
  The cached group keeps the old, destroyed process group.
- `destroy_model_parallel()` does not release these groups.

Failure with reused ranks: `test_sdma_exchange[4]` caches the group
`(0, 1, 2, 3)` on ranks 0 to 3. In
`test_sdma_exchange_under_cfg_parallel[8]`, ranks 0 to 3 use the stale group
and ranks 4 to 7 make a new group. An `ALLGATHER` then times out, and the test
fails after 600 s.

The prototype marks these modules `fresh_ranks`:

- `tests/integration/accelerator/core/distributed/test_sdma_ulysses.py`
- `tests/integration/accelerator/model_executor/models/transformers/test_transformer_wan_sdma.py`
- `tests/integration/accelerator/model_executor/layers/test_fused_qk_norm_rope_wan_flydsl.py`

Suggested fix: make `destroy_model_parallel()` release the SDMA peer groups
of each process group before it destroys the group. The release is
collective (`PeerGroup.close()` uses barriers). Then reset `_slot` and `_call`.
This also matters for a long-lived process that initializes the parallel
state more than once.

## 4. `test_qwen_image21_ulysses` does not restore a module attribute

Location: `tests/integration/accelerator/model_executor/test_qwen_image21_ulysses.py:87`.

```python
usp.get_runtime_state = lambda: SimpleNamespace(attention_backend=AttentionBackendType.SDPA)
```

The worker replaces `xfuser.model_executor.layers.usp.get_runtime_state` and
does not restore it. With reused ranks, the next module
(`test_wan_sequence_parallel_pad.py`) fails:

```text
xfuser/model_executor/layers/usp.py:987 in USP
AttributeError: 'types.SimpleNamespace' object has no attribute 'fp8_comms'
```

Suggested fix: restore the original function in a `finally` block, or use a
context manager that restores it.

## 5. `test_wan_sequence_parallel_pad` patches global attention specs

Location: `tests/integration/accelerator/model_executor/test_wan_sequence_parallel_pad.py`.

- `_attend_in_bf16()` (line 113) and `_attend_as_fp8_comms()` (line 140)
  replace the kernel of a global attention backend spec:
  `object.__setattr__(spec, "_resolved", ...)` (lines 137 and 173). Nothing
  restores the spec.
- The fixture `one_gpu_output` (line 270, `scope="module"`) runs the reference
  with world size 1, so only rank 0 runs it.

With reused ranks, rank 0 has the patched spec and rank 1 does not. In
`test_padded_ulysses_with_fp8_comms_matches_one_gpu[2]`, one rank finished and
waited in `destroy_process_group()` while the other rank was still in the
model forward (`transformer_wan.py:681`). The ranks did different
collectives, and the test hung until I stopped it after 248 s.

Suggested fix: restore `_resolved` after the worker body, for example with a
context manager. Other tests can have the same pattern. Search for
`object.__setattr__(spec` and for assignments to module attributes in the
workers.

## 6. `test_vdn_ulysses_parity` does not run

Location: `tests/integration/accelerator/model_executor/models/transformers/test_vdn_ulysses_parity.py`.

The test spawns 2 ranks (line 146), but its only marker is
`pytest.mark.accelerator` (line 15). Thus:

- `-m "multi_gpu and not nvidia"` does not select it.
- `-m "accelerator and not nvidia and not multi_gpu"` selects it, but on one
  GPU it skips because it needs 2 devices.

Suggested fix: add `pytest.mark.multi_gpu`. Also add a layout check that
rejects a test that calls `accelerator_ranks` with `world_size` greater than 1
and has no `multi_gpu` marker, if such a check is practical.

## 7. Duplicate test module basename

`tests/unit/model_executor/layers/test_quantized_attention_output.py` and
`tests/integration/accelerator/model_executor/layers/test_quantized_attention_output.py`
had the same basename. The documented selectors that collect all test paths
failed at collection. [xdit-project/xDiT#1010](https://github.com/xdit-project/xDiT/pull/1010)
renames the second file to `test_quantized_attention_output_gfx950.py`.
Commit `6d77d2a` on this branch has the same rename. Remove that commit after
PR #1010 merges.

## 8. Reused ranks keep spinning threads

With the rank pool, each rank that ran a test keeps 2 native threads at 100%
CPU, also when the rank is idle. Their thread name is `python`, and their
origin is not known. With the current spawn code, the process exit removes
them. They did not cause a failure, but they use 16 cores on an 8-rank pool.
To identify them, use `py-spy dump --native` in a container started with
`--cap-add SYS_PTRACE`.

## 9. Hangs fail only at the timeout

`spawn_accelerator_ranks` waits for all processes until the timeout (600 s by
default, 1800 s for MOSS). If one rank raises an exception while the other
ranks wait in a collective, the other ranks
can wait until the timeout. I did not
measure this with the current spawn code. With the pool, a test where one
rank raised failed in 5.6 s.

Suggested fix: after the first rank reports an error or exits with a non-zero
code, wait a short time (for example 5 s) for the tracebacks of the other
ranks, then stop them. Give startup and the test body separate timeouts, as
issue #817 suggests.

## Environment notes

- After the node rebooted, `brr status` showed no other jobs, but the brrd
  node snapshots in `/var/log/brr/jobs.jsonl` reported all GPUs as `unknown`.
  brr kept my exclusive job in the queue until I cancelled it.
- The test container did not have `CAP_SYS_PTRACE`, so `py-spy` failed.
  `faulthandler.register(signal.SIGUSR1, all_threads=True)` in each rank, and
  then `kill -USR1 <pid>`, gives Python stacks without ptrace.
- Containers that run `sleep infinity` as PID 1 do not reap zombie processes
  from `docker exec` sessions. This has no effect on the results.
