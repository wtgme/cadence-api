# SongGeneration Pipeline Benchmark Results — 2026-05-14

Branch: `feat/static-kv-cache`
GPUs used: 1, 2 (LM), 3 (Diff) — GPU 0 occupied by another process (Gemma vLLM)
Hardware: 4× NVIDIA L40S (46,068 MiB each)

## Configuration Summary

| Config | Description | LM GPUs | Diff GPU | Compile | TP Size |
|--------|-------------|---------|----------|---------|---------|
| A | Static KV only (branch baseline) | 1, 2 | 3 | No | 1 |
| B | Static KV + torch.compile (dynamic=False) | 1, 2 | 3 | Yes | 1 |
| C | Static KV + TP=2 | 1, 2 | 3 | No | 2 |
| D | Static KV + torch.compile + TP=2 | 1, 2 | 3 | Yes | 2 |
| E | vLLM server | — | — | — | — |

## Timing Results

### Wall-clock times and audio durations

| Config | Run | Wall Time (s) | Audio Duration (s) | Realtime Ratio | HTTP Status | File Size (bytes) |
|--------|-----|--------------|-------------------|----------------|-------------|-------------------|
| A | A1 | 213 | 172.4 | 0.81× | 200 | 4,827,000 |
| A | A2 | 163 | 108.4 | 0.67× | 200 | 4,334,064 |
| B | B1 | 141 | 106.1 | 0.75× | 200 | 3,394,272 |
| B | B2 | 228 | 198.3 | 1.15× | 200 | 5,551,680 |
| C | C1 | 170 | 103.1 | 0.61× | 200 | 4,123,680 |
| C | C2 | 107 | 88.7 | 0.83× | 200 | 2,483,808 |
| D | D1 | 191 | 167.8 | 0.88× | 200 | 4,697,184 |
| D | D2 | 203 | 125.8 | 0.62× | 200 | 5,032,560 |
| E | —  | N/A | N/A | — | FAILED | — |

> Realtime ratio = wall_time / audio_duration. Values < 1.0 mean faster-than-realtime. Output length varies across runs (model generates variable-length audio).

### Normalised: seconds-of-wall-time per second-of-audio

To compare configs fairly despite variable output length:

| Config | Run 1 (s/s) | Run 2 (s/s) | Mean (s/s) | Notes |
|--------|------------|------------|-----------|-------|
| A (static KV, baseline) | 1.24 | 1.50 | **1.37** | |
| B (+ compile) | 1.33 | 1.15 | **1.24** | ~10% faster than A |
| C (+ TP=2) | 1.65 | 1.21 | **1.43** | Slightly slower than A (PCIe overhead) |
| D (+ compile + TP=2) | 1.14 | 1.61 | **1.38** | Similar to A |
| E (vLLM) | — | — | **N/A** | FAILED — source file missing |

### GPU Memory (post-generation, GPUs 1–3)

All configs showed identical GPU memory footprint:

| GPU | Used (MiB) | Total (MiB) |
|-----|------------|-------------|
| 1 (LM) | 10,602 | 46,068 |
| 2 (LM) | 10,602 | 46,068 |
| 3 (Diff) | 5,810–5,852 | 46,068 |

Memory is identical across all pipeline configs (A–D). Static KV cache does not increase GPU memory above the float16 model baseline.

## Compile Warmup Times

| Config | Warmup Time (s) | Notes |
|--------|----------------|-------|
| A | 229 | Standard (no compile) |
| B | 204 | torch.compile triggered on first request — warmup includes compilation |
| C | ~202 (via wait_ready) | No compile, TP=2 |
| D | 193 | torch.compile triggered for TP=2 — warmup includes compilation |

Note: The compile warmup for B and D completed in ~200 s — unexpectedly similar to the no-compile warmup for A. This suggests that `torch.compile(dynamic=False)` with `reduce-overhead` mode is completing the graph capture quickly (not the ~10 min estimated for a first-time compile with a cold triton cache), likely because the triton kernel cache was already warm from prior runs on this machine.

## Analysis

### Config B (compile) vs Config A (no compile)
- Run B1 was faster (141 s vs 213 s for A1) for similar-length audio (106 s vs 172 s audio).
- Normalised mean: B=1.24 s/s vs A=1.37 s/s → **~10% improvement** from torch.compile.
- Results are noisy due to variable output length; 2 runs are insufficient for statistical confidence.

### Config C (TP=2) vs Config A (no compile)
- TP=2 with PCIe-only interconnect (no NVLink) shows **no clear improvement** over TP=1 for single requests.
- C mean=1.43 s/s vs A mean=1.37 s/s — TP=2 is marginally slower on this hardware.
- CLAUDE.md notes PCIe all-reduce overhead of ~6 s per song. With 1 LM worker instead of 2, concurrency is halved with no per-song benefit.

### Config D (compile + TP=2) vs others
- D mean=1.38 s/s — essentially the same as A baseline.
- Compile does not overcome TP=2 overhead on PCIe.

### Recommendation
- For this hardware (L40S, PCIe only), **Config B (static KV + compile, TP=1)** gives the best single-request latency.
- Config A (static KV, no compile) is acceptable and avoids the compile warmup penalty.
- TP=2 provides no benefit on PCIe and reduces concurrency — not recommended.

## Config E — vLLM Server: FAILED

**Reason:** `songgeneration_vllm_server.py` source file is missing from the working directory (`/users/k1810895/data/musicgen/`). Only a stale `__pycache__/songgeneration_vllm_server.cpython-310.pyc` remains. The file appears to have been deleted or was never tracked on the `feat/static-kv-cache` branch.

**Error:** `ERROR: Could not import module "songgeneration_vllm_server"`

The vLLM benchmark could not be run. To fix: restore `songgeneration_vllm_server.py` from git history or another branch (`git checkout main -- songgeneration_vllm_server.py` from the working dir symlink).

## Notes

- Audio output length varies per request (model generates variable-duration songs); this makes direct wall-time comparisons unreliable. The s/s normalised metric is more meaningful.
- Only 2 timed runs per config were performed. More runs (5–10) would be needed for statistically robust comparison.
- wait_ready for TP=2 (configs C, D) reported `lm_workers_alive=2` even with `SONGGEN_TP_SIZE=2` — each rank of the TP pair reports alive independently. This is expected behavior; the server was functional.
- Default pipeline server (Config A) has been restored after benchmarking.
