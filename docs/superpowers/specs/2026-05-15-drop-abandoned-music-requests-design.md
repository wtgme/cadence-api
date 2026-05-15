# Drop Abandoned Music-Generation Requests at the Scheduler

**Status:** Approved
**Date:** 2026-05-15
**Scope:** Server-side change to `songgeneration_pipeline_server.py` only. No Android client changes.

## Problem

When multiple Android clients hit the SongGeneration pipeline API, requests are queued and batched on the server. If a client stops the app (or its in-flight request is cancelled by scene-shift / HR-drift repriming), the TCP connection closes but the server keeps processing the request — wasting GPU compute and, more importantly, taking up a slot in the next `MAX_BATCH_SIZE` batch that an active user could have used. The user-visible symptom is that active users are delayed or fail to hear audio while abandoned requests are draining the worker pool.

## Root Cause

`PipelineScheduler.submit()` puts the new `PendingRequest` into two places:

- `self.pending[rid]` — a dict used by `result_listener` to route chunks back to the correct per-request `asyncio.Queue`.
- `self.request_queue` — an `asyncio.Queue` drained by `scheduler_loop` to form LM batches.

When the client disconnects, the `_stream_request` generator's `finally` block runs and pops `self.pending[rid]`. But the `PendingRequest` object is still sitting in `self.request_queue`. `scheduler_loop` pulls it off, batches it, and dispatches it to an LM worker without checking whether it is still alive. The LM worker generates tokens, diff workers decode chunks, and `result_listener` drops them at the end because `pending[rid]` is gone — by which point all the compute has already happened.

The disconnect signal exists. The scheduler simply does not consume it.

## Design

Three small additions to `PipelineScheduler`, all on the asyncio event loop, no new locks.

### 1. Liveness check at queue-pull time (`scheduler_loop`)

After pulling each request from `self.request_queue` — both the `first` request that starts a batch and each subsequent request collected during the `BATCH_WAIT_MS` window — check whether it is still in `self.pending`. If not, increment `total_abandoned`, log, and `continue`.

```python
async def scheduler_loop(self):
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
        # ... rest unchanged ...
```

### 2. Re-check at dispatch time (`_dispatch_lm`)

Between the moment `scheduler_loop` finalises a batch and the moment `_dispatch_lm` actually writes to the LM worker's `mp.Queue`, more time can pass — especially when the batch waited for `BATCH_WAIT_MS` to fill. Clients can disconnect during that window. Filter the batch one final time inside `_dispatch_lm`, immediately before the `put`:

```python
async def _dispatch_lm(self, batch, lm_idx):
    batch_id = uuid.uuid4().hex[:8]
    live = [pr for pr in batch if pr.request_id in self.pending]
    if not live:
        self.total_dropped_at_dispatch += len(batch)
        log.info("All %d requests in batch went stale — releasing LM worker %d", len(batch), lm_idx)
        self.available_lm.put_nowait(lm_idx)
        return
    if len(live) < len(batch):
        self.total_dropped_at_dispatch += (len(batch) - len(live))
        log.info("Batch shrunk from %d → %d (stale dropped)", len(batch), len(live))
    self._active_lm_batches[batch_id] = lm_idx
    msg = {"batch_id": batch_id, "requests": [pr.params for pr in live]}
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, self.lm_input_queues[lm_idx].put, msg)
```

Two ordering details matter:

- The LM-slot release on an empty-batch path uses `self.available_lm.put_nowait(lm_idx)` so the slot is recycled immediately. Without this, an entirely stale batch would leak an LM slot.
- `_active_lm_batches[batch_id]` is recorded only when the batch is non-empty, so `result_listener` does not later try to release a slot that was never registered.

### 3. Observability — new `/scheduler_stats` endpoint

Add two integer counters to `PipelineScheduler.__init__`:

```python
self.total_abandoned = 0           # skipped pre-batch
self.total_dropped_at_dispatch = 0 # filtered inside _dispatch_lm
```

The existing `/usage` endpoint tails `api_usage.log` and isn't the right place for live counters. Add a new lightweight route alongside it:

```python
@app.get("/scheduler_stats")
def get_scheduler_stats():
    return {
        "total_requests":            scheduler.total_requests,
        "total_batches":             scheduler.total_batches,
        "total_abandoned":           scheduler.total_abandoned,
        "total_dropped_at_dispatch": scheduler.total_dropped_at_dispatch,
    }
```

These let us measure how often the fix fires and decide whether further work (Approach B/C from brainstorming) is ever needed.

## What This Does Not Catch

Once `self.lm_input_queues[lm_idx].put(msg)` succeeds, the LM worker has the batch — we cannot recall it via `mp.Queue`. That batch generates to completion. Chunks come back, `result_listener` looks up `pending[rid]`, finds it popped, and silently drops them (existing behaviour). GPU work for the in-flight batch is still wasted, but the waste is bounded to one batch's runtime instead of the entire pending queue.

This is the accepted trade-off from the brainstorming session: keep the change small, get the dominant benefit (the queue-side leak), defer per-sequence mid-batch cancellation unless metrics show it is still a problem.

## Concurrency Notes

All reads and writes to `self.pending`, `self.request_queue`, `self.available_lm`, `self._active_lm_batches`, and the new counters happen on the asyncio event loop. The multiprocess workers (`lm_worker`, `diff_worker`) only talk via `mp.Queue` objects and never touch `self.pending`. No additional locking is required.

There is no race between `scheduler_loop`'s pop-and-check and `_stream_request`'s `finally`-block pop: both run on the same event loop. Whichever runs first wins, and either outcome is correct (skip-stale, or process-and-drop-late-chunks).

## Testing

Add focused async tests in `tests/test_scheduler_abandoned.py` covering the new scheduler behaviour. They should not require GPU or real LM/diff workers — replace `self.lm_input_queues` with `mp.Queue` instances we can drain in the test, and replace `self.available_lm` / `self.request_queue` with `asyncio.Queue` instances pre-seeded as the real worker startup would have. Use synchronous pytest functions that wrap async bodies with `asyncio.run(...)`, matching the existing `tests/test_static_kv_cache.py` style (no `pytest-asyncio` dependency).

1. **Pre-batch skip.** Submit one request. Pop `pending[rid]` immediately. Run `scheduler_loop` briefly. Assert `lm_input_queues[lm_idx]` received nothing and `total_abandoned == 1`.
2. **Batch shrink at dispatch.** Submit two requests, pop one from `pending`, then call `_dispatch_lm` with both `PendingRequest` objects. Assert the LM queue received a single-request batch with the surviving `request_id` and `total_dropped_at_dispatch == 1`.
3. **Whole-batch stale at dispatch.** Submit two requests. Pop both from `pending` before `_dispatch_lm`. Assert the LM queue received nothing, the LM slot is released back to `available_lm`, `_active_lm_batches` is empty, and `total_dropped_at_dispatch == 2`.

## Files Touched

- `songgeneration_pipeline_server.py` — `PipelineScheduler.__init__`, `PipelineScheduler.scheduler_loop`, `PipelineScheduler._dispatch_lm`, plus a new `/scheduler_stats` route near `/usage`.
- `tests/test_scheduler_abandoned.py` — new test file for scheduler liveness behaviour.

No Android-side changes.

## Out of Scope

- Mid-batch per-sequence cancellation in the LM worker.
- Purging stale entries from `diff_queue`.
- Explicit client cancel endpoint (e.g. `DELETE /cancel/{rid}`) — the existing TCP-disconnect signal is enough for this design.
- Surfacing per-request `rid` to the Android client.

If `/scheduler_stats` later shows a high `total_dropped_at_dispatch` or persistent waste from in-flight batches, revisit by adding Approach B (per-request disconnect watcher) or Approach C (explicit client cancel) from the brainstorming notes.
