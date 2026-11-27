"""Run a multi-GPU test body from pytest without torchrun.

Each rank is a spawned interpreter so CUDA and HIP can initialize. The parent
surfaces the child traceback; a non-zero exit alone hides the failure.
The total timeout includes interpreter startup, imports, and the worker body.
Allow ten minutes by default: cold imports on a slow filesystem took 388 s (#817).

With ``XFUSER_TEST_RANK_POOL=1``, one ``RankPool`` with a rank for each visible
device serves the session. Each rank imports once, then runs each test body that
the parent sends. A test with world size N runs on ranks 0 to N-1. The parent
replaces the pool after a failure or timeout.
"""

import faulthandler
import os
import pickle
import queue
import signal
import sys
import time
import traceback

import pytest

# After one rank fails, wait this long for the tracebacks of the other ranks.
# They often wait in a collective that never completes.
_PEER_GRACE = 5

_pool = None
_pool_starts = 0
# The accelerator conftest sets this for each test that has the ``fresh_ranks`` marker.
fresh_ranks = False


def _require_devices(world_size):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} accelerator devices")
    if not torch.distributed.is_nccl_available():
        pytest.skip("NCCL/RCCL is unavailable")
    return torch


def spawn_accelerator_ranks(worker, tmp_path, *, world_size, timeout=600, init_filename="dist-init", args=()):
    torch = _require_devices(world_size)
    init_method = f"file://{tmp_path / init_filename}"
    if os.environ.get("XFUSER_TEST_RANK_POOL") == "1" and not fresh_ranks:
        _run_in_pool(torch, worker, init_method, world_size, timeout, args)
        return

    context = torch.multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_guard,
            args=(rank, world_size, init_method, worker, result_queue, args),
        )
        for rank in range(world_size)
    ]
    deadline = time.monotonic() + timeout
    for process in processes:
        process.start()
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))

    hung = [process.pid for process in processes if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(5)

    errors = []
    while True:
        try:
            item = result_queue.get(timeout=1)
        except queue.Empty:
            break
        if item is not None:
            errors.append(item)

    exitcodes = [process.exitcode for process in processes]
    if hung or errors or exitcodes != [0] * world_size:
        details = []
        if hung:
            details.append(f"hung pids: {hung}")
        if exitcodes != [0] * world_size:
            details.append(f"exit codes: {exitcodes}")
        details.extend(errors)
        pytest.fail("\n".join(details))


def _guard(rank, world_size, init_method, worker, result_queue, args):
    _set_rank_environment(rank, world_size)
    try:
        worker(rank, world_size, init_method, *args)
    except Exception:
        result_queue.put(traceback.format_exc())
        raise
    result_queue.put(None)


def _set_rank_environment(rank, world_size):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"


def _run_in_pool(torch, worker, init_method, world_size, timeout, args):
    global _pool, _pool_starts
    if _pool is None:
        _pool = RankPool(torch, torch.cuda.device_count())
        _pool_starts += 1
    details = _pool.run(worker, init_method, world_size, args, timeout)
    if details:
        # The ranks can hold broken communicators or a hung collective. Start a new pool for the next test.
        close_pool(timeout=0)
        pytest.fail("\n".join(details))


def close_pool(timeout=10):
    global _pool
    if _pool is not None:
        _pool.close(timeout)
        _pool = None


def pool_starts():
    return _pool_starts


class RankPool:
    """One rank process for each device. The processes run test bodies one after another."""

    def __init__(self, torch, size):
        self.size = size
        context = torch.multiprocessing.get_context("spawn")
        self._tasks = [context.Queue() for _ in range(size)]
        self._results = context.Queue()
        self._processes = [
            context.Process(target=_serve, args=(rank, self._tasks[rank], self._results)) for rank in range(size)
        ]
        for process in self._processes:
            process.start()

    def run(self, worker, init_method, world_size, args, timeout):
        """Run ``worker`` on ranks 0 to ``world_size - 1``. Return the failure details; an empty list means success."""
        payload = pickle.dumps((worker, init_method, args))
        for task_queue in self._tasks[:world_size]:
            task_queue.put((world_size, list(sys.path), payload))
        deadline = time.monotonic() + timeout
        errors = {}
        done = set()
        while len(done) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                rank, error = self._results.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if any(not process.is_alive() for process in self._processes):
                    break
                continue
            done.add(rank)
            if error is not None:
                errors[rank] = error
                deadline = min(deadline, time.monotonic() + _PEER_GRACE)
        details = []
        missing = [rank for rank in range(world_size) if rank not in done]
        alive = [rank for rank in missing if self._processes[rank].is_alive()]
        if alive:
            # Each rank prints the Python stacks of all its threads to stderr.
            for rank in alive:
                os.kill(self._processes[rank].pid, signal.SIGUSR1)
            time.sleep(2)
        if missing:
            exited = {rank: p.exitcode for rank, p in enumerate(self._processes) if not p.is_alive()}
            details.append(f"ranks without a result: {missing}; exited ranks and exit codes: {exited}")
        details.extend(errors[rank] for rank in sorted(errors))
        return details

    def close(self, timeout=10):
        for process, task_queue in zip(self._processes, self._tasks):
            if process.is_alive():
                task_queue.put(None)
        deadline = time.monotonic() + timeout
        for process in self._processes:
            process.join(max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.kill()
                process.join(5)


def _serve(rank, task_queue, result_queue):
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    while True:
        task = task_queue.get()
        if task is None:
            return
        world_size, parent_path, payload = task
        sys.path.extend(entry for entry in parent_path if entry not in sys.path)
        saved_environment = dict(os.environ)
        saved_cwd = os.getcwd()
        _set_rank_environment(rank, world_size)
        error = None
        try:
            worker, init_method, args = pickle.loads(payload)
            worker(rank, world_size, init_method, *args)
        except BaseException:
            error = traceback.format_exc()
        try:
            _reset_rank_state()
        except BaseException:
            error = (error or "") + "state reset failed:\n" + traceback.format_exc()
        os.environ.clear()
        os.environ.update(saved_environment)
        os.chdir(saved_cwd)
        result_queue.put((rank, error))


def _reset_rank_state():
    """Remove the state that one test body can leave for the next."""
    import gc

    import torch

    if "xfuser.core.distributed.parallel_state" in sys.modules:
        from xfuser.core.distributed import parallel_state, runtime_state

        parallel_state.destroy_model_parallel()
        parallel_state.destroy_distributed_environment()
        runtime_state._RUNTIME = None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if "torch._dynamo" in sys.modules:
        torch._dynamo.reset()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
