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


async def _run_scheduler_until_drained(s, deadline_s: float = 0.5):
    """Run scheduler_loop briefly, then cancel. BATCH_WAIT_MS is on the order of
    tens of ms; deadline_s well exceeds it so any pending pulls finish."""
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
