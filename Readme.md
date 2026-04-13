# cadence-api

FastAPI inference servers for the [Cadence](https://github.com/wtgme/cadence) Android app, running on KCL CREATE HPC (GPU cluster). The app generates personalised instrumental music from biometric data via a two-step AI pipeline.

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

- **Cluster:** KCL CREATE HPC — jobs managed via Slurm
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
