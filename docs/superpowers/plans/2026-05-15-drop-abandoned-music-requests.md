# Drop Abandoned Music-Generation Requests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `PipelineScheduler` from dispatching LM batches that contain requests whose clients have already disconnected.

**Architecture:** Add liveness checks in two places in `PipelineScheduler` (`scheduler_loop` pre-batch, `_dispatch_lm` immediately before LM submission), keyed on `self.pending` membership — which the existing `_stream_request` finally block already pops on disconnect. Track two new counters and expose them on a new `/scheduler_stats` HTTP endpoint.

**Tech Stack:** Python 3.10, FastAPI, asyncio, multiprocessing.Queue, pytest (synchronous test functions wrapping `asyncio.run(...)`, matching `tests/test_static_kv_cache.py`).

**Related spec:** `docs/superpowers/specs/2026-05-15-drop-abandoned-music-requests-design.md`

---

## File Structure

- Modify: `songgeneration_pipeline_server.py`
  - `PipelineScheduler.__init__` (around lines 670–685): add two integer counters.
  - `PipelineScheduler.scheduler_loop` (around lines 782–803): skip requests no longer in `self.pending` when pulling from `request_queue`.
  - `PipelineScheduler._dispatch_lm` (around lines 805–810): filter stale requests one final time before writing to the LM worker queue; release the LM slot when the whole batch is stale.
  - Near `@app.get("/usage")` (around line 1095): add a new `@app.get("/scheduler_stats")` route.
- Create: `tests/test_scheduler_abandoned.py` — imports `songgeneration_pipeline_server` and exercises `PipelineScheduler` directly with mock `mp.Queue` instances. No GPU required.

Line numbers are based on the file as of 2026-05-15. If the file shifts, locate the same method bodies.

---

### Task 1: Pre-batch liveness check in `scheduler_loop`

**Files:**
- Modify: `songgeneration_pipeline_server.py` — `PipelineScheduler.__init__`, `PipelineScheduler.scheduler_loop`
- Create: `tests/test_scheduler_abandoned.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scheduler_abandoned.py` with this content:

```python
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
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py::test_pre_batch_skip_abandoned_request -v
```

Expected: FAIL with `AttributeError: 'PipelineScheduler' object has no attribute 'total_abandoned'`.

- [ ] **Step 3: Add the two new counters to `PipelineScheduler.__init__`**

In `songgeneration_pipeline_server.py`, locate the tail of `PipelineScheduler.__init__`:

```python
        self.total_batches   = 0
        self.total_requests  = 0
        self._started        = False
```

Replace with:

```python
        self.total_batches              = 0
        self.total_requests             = 0
        self.total_abandoned            = 0  # skipped pre-batch (client gone before dispatch)
        self.total_dropped_at_dispatch  = 0  # filtered inside _dispatch_lm
        self._started                   = False
```

- [ ] **Step 4: Add the skip logic to `scheduler_loop`**

In `songgeneration_pipeline_server.py`, locate `PipelineScheduler.scheduler_loop`. Current body:

```python
    async def scheduler_loop(self):
        log.info("Scheduler loop started (lm_gpus=%s, diff_gpus=%s, batch_max=%d)",
                 LM_GPU_IDS, DIFF_GPU_IDS, MAX_BATCH_SIZE)
        while True:
            first = await self.request_queue.get()
            batch = [first]
            deadline = asyncio.get_event_loop().time() + BATCH_WAIT_MS / 1000
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=remaining)
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            self.total_batches  += 1
            self.total_requests += len(batch)
            lm_idx = await self.available_lm.get()
            log.info("Dispatching batch of %d to LM worker %d", len(batch), lm_idx)
            asyncio.create_task(self._dispatch_lm(batch, lm_idx))
```

Replace with:

```python
    async def scheduler_loop(self):
        log.info("Scheduler loop started (lm_gpus=%s, diff_gpus=%s, batch_max=%d)",
                 LM_GPU_IDS, DIFF_GPU_IDS, MAX_BATCH_SIZE)
        while True:
            first = await self.request_queue.get()
            if first.request_id not in self.pending:
                self.total_abandoned += 1
                log.info("Skipping abandoned request %s (pre-batch)", first.request_id)
                continue
            batch = [first]
            deadline = asyncio.get_event_loop().time() + BATCH_WAIT_MS / 1000
            while len(batch) < MAX_BATCH_SIZE:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if req.request_id not in self.pending:
                    self.total_abandoned += 1
                    log.info("Skipping abandoned request %s (pre-batch)", req.request_id)
                    continue
                batch.append(req)

            self.total_batches  += 1
            self.total_requests += len(batch)
            lm_idx = await self.available_lm.get()
            log.info("Dispatching batch of %d to LM worker %d", len(batch), lm_idx)
            asyncio.create_task(self._dispatch_lm(batch, lm_idx))
```

Two non-obvious points:
- The skip check for the first request happens before `batch = [first]`, so an abandoned first request goes straight back to the `while True:` top without paying for the BATCH_WAIT_MS coalescing window.
- The skip check for additional requests has been moved out of the `try` block so an abandoned sibling does not get `append`'d to the batch.

- [ ] **Step 5: Run the test, verify it passes**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py::test_pre_batch_skip_abandoned_request -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /users/k1810895/data/musicgen
git add songgeneration_pipeline_server.py tests/test_scheduler_abandoned.py
git commit -m "Skip abandoned requests at scheduler queue pull"
```

---

### Task 2: Dispatch-time re-check in `_dispatch_lm`

**Files:**
- Modify: `songgeneration_pipeline_server.py` — `PipelineScheduler._dispatch_lm`
- Modify: `tests/test_scheduler_abandoned.py`

- [ ] **Step 1: Append two more failing tests**

Add the following to the bottom of `tests/test_scheduler_abandoned.py`:

```python
def test_dispatch_filter_drops_one_of_two():
    """One of two batched requests goes stale between queue-pull and dispatch."""
    async def run():
        s = _make_scheduler()
        rid1, _ = await s.submit({"lyric": "."}, client_ip="t1")
        rid2, _ = await s.submit({"lyric": "."}, client_ip="t2")
        pr1 = s.pending[rid1]
        pr2 = s.pending[rid2]
        # Pop one from pending — simulating disconnect after queue-pull,
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
    """All requests in a batch go stale before dispatch — LM slot is recycled."""
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
```

- [ ] **Step 2: Run all tests, verify the two new ones fail**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py -v
```

Expected: `test_pre_batch_skip_abandoned_request` PASS; the two new tests FAIL — `total_dropped_at_dispatch` stays at 0 and the stale batch is dispatched in full.

- [ ] **Step 3: Add the dispatch-time filter to `_dispatch_lm`**

In `songgeneration_pipeline_server.py`, locate `_dispatch_lm`. Current body:

```python
    async def _dispatch_lm(self, batch: list[PendingRequest], lm_idx: int):
        batch_id = uuid.uuid4().hex[:8]
        self._active_lm_batches[batch_id] = lm_idx  # so result_listener can release the slot
        msg = {"batch_id": batch_id, "requests": [pr.params for pr in batch]}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.lm_input_queues[lm_idx].put, msg)
```

Replace with:

```python
    async def _dispatch_lm(self, batch: list[PendingRequest], lm_idx: int):
        batch_id = uuid.uuid4().hex[:8]
        live = [pr for pr in batch if pr.request_id in self.pending]
        if not live:
            # Whole batch went stale between queue-pull and dispatch.
            # Recycle the LM slot and do not register a phantom batch.
            self.total_dropped_at_dispatch += len(batch)
            log.info("All %d requests in batch went stale — releasing LM worker %d",
                     len(batch), lm_idx)
            self.available_lm.put_nowait(lm_idx)
            return
        if len(live) < len(batch):
            self.total_dropped_at_dispatch += (len(batch) - len(live))
            log.info("Batch shrunk from %d → %d (stale dropped)", len(batch), len(live))
        self._active_lm_batches[batch_id] = lm_idx  # so result_listener can release the slot
        msg = {"batch_id": batch_id, "requests": [pr.params for pr in live]}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.lm_input_queues[lm_idx].put, msg)
```

Ordering note: `_active_lm_batches[batch_id]` is recorded **after** the empty-batch check so an empty batch does not leave a stale entry; `result_listener` only looks up keys it expects to find.

- [ ] **Step 4: Run all tests, verify they pass**

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /users/k1810895/data/musicgen
git add songgeneration_pipeline_server.py tests/test_scheduler_abandoned.py
git commit -m "Filter abandoned requests at LM dispatch and recycle slot when batch is fully stale"
```

---

### Task 3: Expose counters via `/scheduler_stats` endpoint

**Files:**
- Modify: `songgeneration_pipeline_server.py` — add new route near `/usage`

- [ ] **Step 1: Add the new endpoint**

In `songgeneration_pipeline_server.py`, locate the `@app.get("/usage")` route. Immediately after its function body, add:

```python
@app.get("/scheduler_stats")
def get_scheduler_stats():
    """Live counters from the PipelineScheduler.

    Useful for confirming abandoned-request handling is firing in production.
    Unlike /usage (which tails api_usage.log), this returns the in-memory
    counters maintained by scheduler_loop and _dispatch_lm."""
    return {
        "total_requests":            scheduler.total_requests,
        "total_batches":             scheduler.total_batches,
        "total_abandoned":           scheduler.total_abandoned,
        "total_dropped_at_dispatch": scheduler.total_dropped_at_dispatch,
    }
```

- [ ] **Step 2: Smoke-test the endpoint**

Per CLAUDE.md the user manages the Slurm job and restarts the server; do not attempt to restart it yourself. Ask the user to restart the API service, then, with the SSH tunnel up, from the dev machine run:

```bash
curl -s http://localhost:8888/scheduler_stats | python -m json.tool
```

Expected output (all zeros immediately after restart):

```json
{
    "total_requests": 0,
    "total_batches": 0,
    "total_abandoned": 0,
    "total_dropped_at_dispatch": 0
}
```

- [ ] **Step 3: Commit**

```bash
cd /users/k1810895/data/musicgen
git add songgeneration_pipeline_server.py
git commit -m "Expose scheduler counters via /scheduler_stats endpoint"
```

---

## Verification

After all three tasks complete, run the full new test file once more to confirm nothing regressed:

```bash
cd /users/k1810895/data/musicgen
conda run -n musicgen python -m pytest tests/test_scheduler_abandoned.py -v
```

Expected: 3 passed.

Manual end-to-end check (requires the user to restart the API service):

1. Start a `/generate_stream` request from the Android app or with `curl`.
2. Disconnect early (`Ctrl-C` for curl, or stop the Cadence app).
3. Hit `curl http://localhost:8888/scheduler_stats` — `total_abandoned` should be at least 1, or `total_dropped_at_dispatch` should be at least 1.
