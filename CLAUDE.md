# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

Three inference servers, all using the `musicgen` conda env (Python 3.10):

| Server | File | Port | Model | GPUs | Conda env |
|---|---|---|---|---|---|
| SongGeneration Pipeline (**default**) | `songgeneration_pipeline_server.py` | 8888 | `SongGeneration-v2-large` (4B params) | auto-split LM+Diff, any N≥1 | `musicgen` (Python 3.10) |
| SongGeneration vLLM (legacy) | `songgeneration_vllm_server.py` | 8888 | `SongGeneration-v2-large` (4B params) | multi-GPU batched | `musicgen` (Python 3.10) |
| SongGeneration (legacy) | `songgeneration_server.py` | 8888 | `SongGeneration-v2-large` (4B params) | single GPU | `musicgen` (Python 3.10) |
| HeartMuLa | `heartmula_server.py` | 8888 | `HeartMuLa-oss-3B-happy-new-year` (3B params) | single GPU | `musicgen` (Python 3.10) |

**All servers expose identical REST APIs on port 8888 — only one runs at a time.** The Pipeline server is the default deployment; it auto-detects available GPUs, splits them between LM workers (batched autoregressive) and Diffusion workers (parallel decode), and streams MP3 chunks with ~40s TTFB for first audio. The Android app does not need changes — same base URL.

## Starting the servers

```bash
# SongGeneration Pipeline (port 8888) — DEFAULT (auto GPU split, streaming, ~40s TTFB)
# Prefer start_songgen_pipeline.sh which handles GPU assignment, auto-restart, and optimisation flags.
bash /cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen/start_songgen_pipeline.sh
# Optimisations active: torch.compile (SONGGEN_COMPILE=1), Flash-Decoding (always on), Flash Attention 2 (always on).
# First request after startup is slow (~extra 2–3 min) while torch.compile JITs the LM kernels.

# SongGeneration vLLM (port 8888) — batched only, no streaming
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && SONGGEN_GPU_IDS=0,1 SONGGEN_BATCH_MAX=2 conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_vllm_server:app --host 0.0.0.0 --port 8888" >> logs/vllm_stdout.log 2>&1 &

# SongGeneration legacy single-GPU (port 8888)
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_server:app --host 0.0.0.0 --port 8888" >> logs/songgen_stdout.log 2>&1 &

# HeartMuLa (port 8888) — alternative model
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && conda run -n musicgen --no-capture-output python -m uvicorn heartmula_server:app --host 0.0.0.0 --port 8888" >> logs/heartmula_stdout.log 2>&1 &
```

### Pipeline server environment variables

| Variable | Default | Description |
|---|---|---|
| `SONGGEN_LM_GPU_IDS` | auto | Comma-separated GPU ids for LM workers (leave unset for auto) |
| `SONGGEN_DIFF_GPU_IDS` | auto | Comma-separated GPU ids for Diff workers (leave unset for auto) |
| `SONGGEN_BATCH_MAX` | `2` | Max requests per LM batch |
| `SONGGEN_BATCH_WAIT_MS` | `500` | Max ms to wait for batch to fill |
| `SONGGEN_COMPILE` | `0` | Set to `1` to enable `torch.compile(reduce-overhead)` on the LM — ~20–35% faster per-token after warm-up; first request takes several extra minutes to compile |
| `SONGGEN_TP_SIZE` | `1` | GPUs per LM worker for tensor parallelism. Set to `2` to split each LM worker across 2 GPUs via DTensor (ColwiseParallel + RowwiseParallel on all attention/FFN layers). ~1.7× faster per-song; fewer concurrent workers. Flash Attention 2 is automatically disabled for TP. |
| `SONGGEN_TP_BASE_PORT` | `29500` | Base TCP port for NCCL rendezvous between TP rank pairs. Each TP worker pair uses `base_port + worker_id * 10`. |

Auto-split: both vars must be set together or both left unset.

**TP_SIZE=1 (default):** `diff = max(1, N//6)`, `lm = N - diff`

| N GPUs | LM workers | Diff workers |
|---|---|---|
| 1 | 1 (shared) | 1 (shared) |
| 2 | 1 | 1 |
| 4 | 3 | 1 |
| 8 | 7 | 1 |

**TP_SIZE=2:** `n_lm_workers = (N-1)//2`, `diff = N - n_lm_workers*2`

| N GPUs | LM workers (×2 GPU each) | Diff workers |
|---|---|---|
| 4 | 1 | 2 |
| 6 | 2 | 2 |
| 8 | 3 | 2 |

**Trade-off:** TP=2 reduces per-song latency ~1.7× but halves the number of concurrent songs. Best for low-concurrency use cases (e.g. Android app with one user at a time). GPU interconnect is PCIe NODE (no NVLink); all-reduce overhead is ~6 s per song (~3%) — acceptable.

**Why 1 diff worker is enough (TP=1):** diff decodes 1 chunk in ~1.8 s; in the ~3 min an LM takes it processes ~100 chunks — capacity for ~16 concurrent LM workers.

**Memory during startup:** each worker loads `model.pt` (13 GB, mmap) + `model_septoken` (4.5 GB) — peak ~16 GB CPU RAM per worker. Workers load **sequentially** (each waits for "ready" before the next starts), so total peak ≈ 16 GB + N × 0.5 GB ≈ 20 GB for 8 GPUs. An **80 GB job is sufficient**. Loading simultaneously would require N × 16 GB and OOM at 80 GB with ≥6 GPUs.

**Performance optimisations already active:** Flash Attention 2, cuDNN auto-tune (benchmark=True), TF32 matmul (allow_tf32=True), float16 weights, CUDA cache cleared after every decode.

> Do NOT use `python -m uvicorn` from system Python or activate the env first — use `conda run -n musicgen`.
> The system `uvicorn` binary is broken. Conda must be sourced via full path (see above).

## Checking status / logs

```bash
curl http://localhost:8888/health   # whichever server is running (vLLM shows GPU/queue stats)

tail -f logs/vllm_server.log        # vLLM server structured log
tail -f logs/vllm_stdout.log        # vLLM server stdout
tail -f logs/songgen_stdout.log     # legacy server stdout
tail -f logs/heartmula_stdout.log

# Recent API usage
curl http://localhost:8888/usage
```

## SongGeneration API (`songgeneration_server.py`, port 8888)

`POST /generate` — JSON body → MP3 audio bytes

```json
{
  "lyric": ".",
  "descriptions": "ambient,calm,peaceful,soft",
  "auto_prompt_audio_type": "Soundtrack",
  "generate_type": "bgm"
}
```

**Fields:**
- `lyric` — ignored for `bgm` mode (server replaces with `"."` internally). Pass `"."`.
- `descriptions` — comma-separated lowercase style tags. Server prepends `[Musicality-very-high], [Pure-Music], ` for bgm.
- `auto_prompt_audio_type` — optional reference style: `Pop, Latin, Rock, Electronic, Metal, Country, R&B/Soul, Ballad, Jazz, World, Hip-Hop, Funk, Soundtrack, Auto`
- `generate_type` — `bgm` (instrumental only, default), `mixed` (vocals+bgm), `vocal`

**Style controls for `bgm`:** only `descriptions` and `auto_prompt_audio_type` affect the output. Lyrics have no effect.

`GET /health` — liveness + model loaded status  
`GET /usage?n=50` — tail recent API usage log

## HeartMuLa API (`heartmula_server.py`, port 8888)

Same REST interface as SongGeneration — Android app can switch servers without changing the base URL.

`POST /generate` — JSON body → MP3 audio bytes

```json
{
  "lyric": ".",
  "descriptions": "ambient,calm,peaceful,soft",
  "auto_prompt_audio_type": "Soundtrack",
  "generate_type": "bgm"
}
```

**Fields:**
- `lyric` — ignored for `bgm` mode. Pass `"."`.
- `descriptions` — comma-separated lowercase style tags (maps to HeartMuLa's `tags`).
- `auto_prompt_audio_type` — appended to tags as a lowercase genre hint.
- `generate_type` — `bgm` (default) or `mixed`.
- `max_audio_length_ms` — default 120000 (2 min).
- `temperature`, `topk`, `cfg_scale` — generation parameters.

**Key paths:**
- heartlib: `heartmula/heartlib/src/` (added to sys.path at startup)
- Checkpoint: `heartmula/ckpt/HeartMuLa-oss-3B/` + `heartmula/ckpt/HeartCodec-oss/`
- gen_config + tokenizer: `heartmula/gen/` (symlinked into `heartmula/ckpt/` at startup)

## Architecture: SongGeneration

Hybrid LLM-Diffusion model (LeVo 2 / CodecLM). Two-stage pipeline:
1. **LM stage** — 4B-param transformer generates discrete audio tokens conditioned on descriptions + style reference
2. **Diffusion stage** — separate tokenizer (diffusion-based audio codec) decodes tokens to waveform

Startup loads: auto-prompt library (`tools/new_auto_prompt.pt`), separate tokenizer, LM (float16).  
GPU memory: ~22 GB. Serial GPU access enforced via `asyncio.Semaphore(1)`.  
Generation timeout: 300s. OOM recovery: `gc.collect()` + `torch.cuda.empty_cache()`.

**Key paths:**
- Repo: `/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen/songgeneration/`
- Checkpoint: `ckpt/songgeneration_v2_large/` (`config.yaml` + `model.pt`)
- Auto-prompt library: `tools/new_auto_prompt.pt`
- HuggingFace: `lglg666/SongGeneration-v2-large`

## Architecture: HeartMuLa

`HeartMuLaGenPipeline` (heartlib). Two-stage pipeline:
1. **HeartMuLa LM** — 3B-param transformer generates discrete audio tokens from lyrics + tags
2. **HeartCodec** — 12.5 Hz music codec decodes tokens to 48 kHz waveform

GPU memory: ~16 GB. Uses same `asyncio.Semaphore(1)` serial access pattern.  
HuggingFace: `HeartMuLa/HeartMuLa-oss-3B-happy-new-year` + `HeartMuLa/HeartCodec-oss-20260123`

## Conda environments

| Env | Python | torch | Used for |
|---|---|---|---|
| `musicgen` | 3.10 | 2.6.0+cu124 | SongGeneration + HeartMuLa servers |

## Logs

All logs in `logs/`:
- `songgen_stdout.log` — SongGeneration server stdout
- `server.log` — structured SongGeneration server log (rotating, 10MB)
- `api_usage.log` — per-request JSON usage log
- `heartmula_stdout.log` — HeartMuLa server stdout
- `heartmula_server.log` — structured HeartMuLa server log
- `heartmula_api_usage.log` — per-request JSON usage log
