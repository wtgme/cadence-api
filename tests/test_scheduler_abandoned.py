"""
Tests for PipelineScheduler abandoned-request handling.

Run with:
  cd /users/k1810895/data/musicgen
  conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py -v
"""
import asyncio
import multiprocessing as mp
import sys
from pathlib import Path

ROOT = Path("/users/k1810895/data/musicgen")
sys.path.insert(0, str(ROOT / "songgeneration"))
sys.path.insert(0, str(ROOT / "songgeneration" / "codeclm" / "tokenizer"))
sys.path.insert(0, str(ROOT))

import songgeneration_pipeline_server as ps


def _make_scheduler() -> "ps.PipelineScheduler":
    """Build a PipelineScheduler with mock LM queues and pre-filled available_lm.
    No real workers are started — only the asyncio bookkeeping is exercised."""
    s = ps.PipelineScheduler()
    s.lm_input_queues = [mp.Queue()]
    s.available_lm    = asyncio.Queue()
    s.request_queue   = asyncio.Queue()
    s.available_lm.put_nowait(0)
    return s


async def _run_scheduler_until_drained(s, deadline_s: float = 0.2):
    """Run scheduler_loop briefly, then cancel.

    The test cases here put the scheduler into the abandoned-skip path
    immediately — the loop pulls one entry, finds it not in `pending`,
    and continues. We never enter the coalescing window, so deadline_s
    only needs to be long enough for that one pull-and-skip iteration."""
    task = asyncio.create_task(s.scheduler_loop())
    await asyncio.sleep(deadline_s)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


def test_pre_batch_skip_abandoned_request():
    async def run():
        s = _make_scheduler()
        rid, _cq = await s.submit({"lyric": "."}, client_ip="test")
        # Simulate _stream_request finally: client disconnected.
        s.pending.pop(rid)
        await _run_scheduler_until_drained(s)
        assert s.total_abandoned == 1, f"expected 1 abandoned, got {s.total_abandoned}"
        # Mock LM queue must be empty — request must never have been dispatched.
        assert s.lm_input_queues[0].empty(), "stale request was dispatched"
        # LM slot must still be available.
        assert s.available_lm.qsize() == 1
    asyncio.run(run())


def test_dispatch_filter_drops_one_of_two():
    """One of two batched requests goes stale between queue-pull and dispatch."""
    async def run():
        s = _make_scheduler()
        rid1, _ = await s.submit({"lyric": "."}, client_ip="t1")
        rid2, _ = await s.submit({"lyric": "."}, client_ip="t2")
        pr1 = s.pending[rid1]
        pr2 = s.pending[rid2]
        # Pop one from pending -- simulating disconnect after queue-pull,
        # before _dispatch_lm runs.
        s.pending.pop(rid1)
        # Acquire an LM slot (matches what scheduler_loop would do).
        lm_idx = await s.available_lm.get()
        await s._dispatch_lm([pr1, pr2], lm_idx)
        assert s.total_dropped_at_dispatch == 1
        msg = s.lm_input_queues[0].get(timeout=1)
        assert len(msg["requests"]) == 1
        assert msg["requests"][0]["request_id"] == rid2
    asyncio.run(run())


def test_dispatch_filter_drops_whole_batch():
    """All requests in a batch go stale before dispatch -- LM slot is recycled."""
    async def run():
        s = _make_scheduler()
        rid1, _ = await s.submit({"lyric": "."}, client_ip="t1")
        rid2, _ = await s.submit({"lyric": "."}, client_ip="t2")
        pr1 = s.pending[rid1]
        pr2 = s.pending[rid2]
        s.pending.pop(rid1)
        s.pending.pop(rid2)
        lm_idx = await s.available_lm.get()
        assert s.available_lm.qsize() == 0
        await s._dispatch_lm([pr1, pr2], lm_idx)
        assert s.total_dropped_at_dispatch == 2
        # Nothing dispatched to the LM worker.
        assert s.lm_input_queues[0].empty()
        # LM slot was returned to the pool.
        assert s.available_lm.qsize() == 1
        # No phantom batch was registered.
        assert s._active_lm_batches == {}
    asyncio.run(run())
