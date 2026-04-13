# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Overview

Two inference servers running on the same GPU machine (erc-hpc-comp204), both using the `musicgen` conda env (Python 3.10):

| Server | File | Port | Model | Conda env |
|---|---|---|---|---|
| SongGeneration (**default**) | `songgeneration_server.py` | 8000 | `SongGeneration-v2-large` (4B params) | `musicgen` (Python 3.10) |
| HeartMuLa | `heartmula_server.py` | 8000 | `HeartMuLa-oss-3B-happy-new-year` (3B params) | `musicgen` (Python 3.10) |

**Both servers expose identical REST APIs on port 8000 — only one runs at a time.** SongGeneration is the default deployment; HeartMuLa is an alternative that can be swapped in by restarting on port 8000. The Android app switches between them by redeploying the server (no client-side URL change needed).

## Starting the servers

```bash
# SongGeneration (port 8000) — DEFAULT
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_server:app --host 0.0.0.0 --port 8000" >> logs/songgen_stdout.log 2>&1 &

# HeartMuLa (port 8000) — alternative deployment (stop SongGeneration first)
nohup bash -c "source /software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh && cd /users/k1810895/data/musicgen && conda run -n musicgen --no-capture-output python -m uvicorn heartmula_server:app --host 0.0.0.0 --port 8000" >> logs/heartmula_stdout.log 2>&1 &
```

> Do NOT use `python -m uvicorn` from system Python or activate the env first — use `conda run -n musicgen`.
> The system `uvicorn` binary is broken. Conda must be sourced via full path (see above).

## Checking status / logs

```bash
curl http://localhost:8000/health   # whichever server is running

tail -f logs/songgen_stdout.log
tail -f logs/heartmula_stdout.log

# Recent API usage
curl http://localhost:8000/usage
```

## SongGeneration API (`songgeneration_server.py`, port 8000)

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

## HeartMuLa API (`heartmula_server.py`, port 8000)

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
