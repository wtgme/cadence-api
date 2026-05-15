# cadence-api

FastAPI inference servers for the [Cadence](https://github.com/wtgme/cadence) Android app, running on a GPU HPC cluster. The app generates personalised instrumental music from biometric data via a two-step AI pipeline.

## Servers

| File | Model | Role |
|------|-------|------|
| `songgeneration_server.py` | `lglg666/SongGeneration-v2-large` | Lyrics + style tags → MP3 (primary music backend) |
| `heartmula_server.py` | `HeartMuLa-oss-3B-happy-new-year` | Alternative lyrics-conditioned music generation |
| `gemma_server.py` | `google/gemma-4-26B-A4B-it` | Multimodal LLM for signal interpretation (text + vision) |

All three servers expose a compatible REST API so the Android app can swap backends by changing a single URL.

## API Endpoints (all servers)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Generate audio from lyrics + descriptions (returns `audio/mpeg`) |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/usage` | Tail recent API usage log entries |

`gemma_server.py` additionally exposes `POST /generate-vision` for image-conditioned text generation (multipart form).

### `/generate` request body

```json
{
  "lyric": "verse and chorus text...",
  "descriptions": ["Electronic", "Upbeat", "120bpm"],
  "auto_prompt_audio_type": "Pop",
  "generate_type": "bgm"
}
```

`generate_type`: `bgm` (instrumental), `mixed` (vocals + accompaniment), `vocal`.

## Dependencies

### Model weights (not in git — must exist on the HPC filesystem)

| Directory | Contents | Size |
|-----------|----------|------|
| `songgeneration/` | SongGeneration-v2-large checkpoint + `codeclm` tokenizer/codec packages | ~27 GB |
| `heartmula/` | HeartMuLa-oss-3B checkpoint + `heartlib` Python package | ~21 GB |

The servers import `codeclm` and `heartlib` directly from these directories (added to `sys.path` at startup) — no separate pip install needed.

### Python packages (conda env: `musicgen`, Python 3.10)

```
fastapi
uvicorn
pydantic
torch
torchaudio
numpy
omegaconf
transformers
bitsandbytes
Pillow
```

## Infrastructure

- **Cluster:** HPC cluster — jobs managed via Slurm
- **GPU:** L40S (48 GB VRAM)
- **Working dir:** `/users/k1810895/data/musicgen`
- **Conda env:** `musicgen` (Python 3.10)
- **Logs:** `logs/` (excluded from git)

## Accessing from the Android app

The compute node is not publicly reachable. The Cadence app connects via an SSH tunnel:

```
localhost:8888  →  <compute-node>:37629   (SongGeneration)
```

See `scripts/hpc-tunnel.sh` in the [cadence](https://github.com/wtgme/cadence) repo for tunnel management.

## References

- SongGeneration v2-large: https://arxiv.org/pdf/2506.07520
- HeartMuLa: https://arxiv.org/pdf/2601.10547

## SongGeneration inference optimizations

The upstream LM decode loop in `songgeneration/codeclm/models/llama/modeling_llama.py`
ran `torch.cat` on the KV cache at every step — O(N²) total memory bandwidth
over ~6,750 decode steps × 36 layers × CFG batch. This made a 270 s song take
~6 min 11 s. Three changes in this fork bring it to ~1.6× faster:

1. **Static KV cache.** `LlamaAttention` and `LlamaFlashAttention2` now accept
   a `cache_position: int`. When set, K/V are written **in place** into a
   pre-allocated `[B, max_seq, H, D]` buffer instead of growing via `torch.cat`.
   Buffers are allocated lazily after prefill in `LmModel._init_static_cache`
   (`songgeneration/codeclm/models/lm_levo.py`) and live in `_streaming_state`
   across decode calls.

2. **Flash-Decoding.** The decode path in `LlamaFlashAttention2` replaces the
   write → slice → transpose → `flash_attn_func` sequence with a single fused
   `flash_attn_with_kvcache` call. This uses the Flash-Decoding kernel
   (parallelises the KV read across SMs) and writes the new K/V into the cache
   in-place inside the kernel — eliminating the host-side slice and transpose.

3. **`torch.compile(dynamic=False)`** in `songgeneration_pipeline_server.py`.
   With fixed-shape decode steps, the compiler can capture and replay a CUDA
   graph, eliminating Python dispatch overhead. Enable with `SONGGEN_COMPILE=1`
   (first request pays a ~3–10 min warmup; subsequent requests are fast).

Measured on L40S PCIe (LM on GPUs 1,2; diff on GPU 3):

| Config                                | Wall-s / audio-s | Speedup vs upstream |
|---------------------------------------|------------------|---------------------|
| Upstream (growing KV cache)           | 1.37             | 1.00×               |
| + static KV cache only                | 1.37             | 1.00×               |
| + static KV cache + `torch.compile`   | 1.24             | 1.10×               |
| + Flash-Decoding                      | 0.94             | 1.46×               |
| **+ Flash-Decoding + `torch.compile`**| **0.84**         | **1.63×**           |

Full design notes and benchmark methodology are in
`docs/superpowers/specs/2026-05-13-static-kv-cache-design.md`,
`docs/superpowers/plans/2026-05-13-static-kv-cache.md`, and
`docs/benchmark-results-2026-05-14.md`.

## Acknowledgments

The `songgeneration/` directory contains source code derived from
[tencent-ailab/songgeneration](https://github.com/tencent-ailab/songgeneration)
(Apache 2.0). See `songgeneration/LICENSE` for the original license.
