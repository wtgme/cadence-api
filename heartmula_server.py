"""
HeartMuLa-oss-3B — FastAPI inference server (stateless, simplified).

Same REST API as songgeneration_server_simple.py so the Android app can
switch backends by changing a single URL.

Endpoints
---------
POST /generate  — lyrics + descriptions → song (JSON body, returns audio/mpeg)
GET  /health    — liveness check
GET  /usage     — tail recent API usage log entries
"""

import asyncio
import gc
import io
import json
import logging
import logging.handlers
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/users/k1810895/data/musicgen/heartmula")

# The ckpt dir must have this structure (from README):
#   ckpt/
#   ├── HeartCodec-oss/           ← HeartMuLa/HeartCodec-oss-20260123
#   ├── HeartMuLa-oss-3B/         ← HeartMuLa/HeartMuLa-oss-3B-happy-new-year
#   ├── gen_config.json           ← HeartMuLa/HeartMuLaGen
#   └── tokenizer.json            ← HeartMuLa/HeartMuLaGen
CKPT_DIR = BASE_DIR / "ckpt"

# heartlib installed from BASE_DIR/heartlib
HEARTLIB_SRC = BASE_DIR / "heartlib" / "src"
if str(HEARTLIB_SRC) not in sys.path:
    sys.path.insert(0, str(HEARTLIB_SRC))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path("/users/k1810895/data/musicgen/logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "heartmula_server.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
log = logging.getLogger(__name__)

usage_logger = logging.getLogger("usage")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False
_usage_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "heartmula_api_usage.log", maxBytes=10 * 1024 * 1024, backupCount=10
)
_usage_handler.setFormatter(logging.Formatter("%(message)s"))
usage_logger.addHandler(_usage_handler)


def log_usage(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    usage_logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Model globals
# ---------------------------------------------------------------------------
MODEL_ID = "HeartMuLa-oss-3B-happy-new-year"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_VERSION = "3B"

app = FastAPI(title="HeartMuLa-oss-3B API", version="1.0.0")

_gpu_sem: asyncio.Semaphore | None = None
_pipe = None
_ready = False


def _ensure_ckpt_layout():
    """
    gen_config.json and tokenizer.json were downloaded to BASE_DIR/gen/.
    The pipeline expects them at CKPT_DIR root. Symlink them if needed.
    """
    gen_dir = BASE_DIR / "gen"
    for fname in ("gen_config.json", "tokenizer.json"):
        src = gen_dir / fname
        dst = CKPT_DIR / fname
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
            log.info("Symlinked %s → %s", src, dst)


@app.on_event("startup")
async def load_model():
    global _pipe, _ready, _gpu_sem

    _gpu_sem = asyncio.Semaphore(1)

    _ensure_ckpt_layout()

    from heartlib import HeartMuLaGenPipeline

    log.info("Loading %s from %s on %s…", MODEL_ID, CKPT_DIR, DEVICE)
    t0 = time.time()

    _pipe = HeartMuLaGenPipeline.from_pretrained(
        str(CKPT_DIR),
        device={
            "mula": torch.device(DEVICE),
            "codec": torch.device(DEVICE),
        },
        dtype={
            "mula": torch.bfloat16,
            "codec": torch.float32,
        },
        version=MODEL_VERSION,
        lazy_load=False,
    )

    gc.collect()
    torch.cuda.empty_cache()
    _ready = True
    log.info("Model fully ready in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


def _run_generation(
    lyric: str,
    descriptions: Optional[str],
    generate_type: str,
    max_audio_length_ms: int,
    temperature: float,
    topk: int,
    cfg_scale: float,
) -> bytes:
    """Blocking generation — runs in thread executor to avoid blocking the event loop."""

    # For bgm mode, lyrics are ignored (pure instrumental).
    # Pass a minimal placeholder; tags drive the style entirely.
    effective_lyric = "." if generate_type == "bgm" else lyric

    # tags: use descriptions as-is (comma-separated lowercase style tags)
    tags = descriptions if descriptions else "ambient,calm"

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with torch.no_grad():
            _pipe(
                {"lyrics": effective_lyric, "tags": tags},
                max_audio_length_ms=max_audio_length_ms,
                save_path=tmp_path,
                topk=topk,
                temperature=temperature,
                cfg_scale=cfg_scale,
            )
        with open(tmp_path, "rb") as f:
            return f.read()
    except RuntimeError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        raise
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    lyric: str = Field(..., description=(
        "Song lyrics. Ignored when generate_type='bgm' (instrumental). "
        "For bgm, pass '.' or any placeholder."
    ))
    descriptions: Optional[str] = Field(None, description=(
        "Comma-separated style tags, e.g. 'electronic,energetic,synthesizer'. "
        "These map to HeartMuLa's 'tags' field."
    ))
    auto_prompt_audio_type: Optional[str] = Field(None, description=(
        "Style hint appended to tags if provided, e.g. 'Electronic', 'Jazz'. "
        "HeartMuLa uses tag-only conditioning so this is merged into descriptions."
    ))
    generate_type: str = Field("bgm", description=(
        "Output type: 'bgm' (instrumental, default), 'mixed' (vocals+accompaniment)."
    ))
    max_audio_length_ms: int = Field(120_000, description="Maximum audio length in milliseconds.")
    temperature: float = Field(1.0, description="Sampling temperature.")
    topk: int = Field(50, description="Top-k sampling parameter.")
    cfg_scale: float = Field(1.5, description="Classifier-free guidance scale.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _ready,
    }


@app.get("/usage", summary="Recent API usage log entries")
def get_usage(n: int = 50):
    usage_log = LOG_DIR / "heartmula_api_usage.log"
    if not usage_log.exists():
        return []
    lines = usage_log.read_text().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return entries


@app.post(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}},
    summary="Lyrics + tags → song (bgm = pure instrumental)",
)
async def generate(req: GenerateRequest, request: Request):
    if not _ready:
        raise HTTPException(503, "Model not loaded yet.")
    if req.generate_type not in ("mixed", "bgm"):
        raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}. Use 'bgm' or 'mixed'.")

    # Merge auto_prompt_audio_type into tags so HeartMuLa picks up the style hint
    tags = req.descriptions or ""
    if req.auto_prompt_audio_type:
        genre_tag = req.auto_prompt_audio_type.lower().replace("/", "-").replace("&", "and")
        if genre_tag not in tags.lower():
            tags = f"{tags},{genre_tag}".lstrip(",")

    ip = client_ip(request)
    log.info(
        "[%s] /generate lyric_len=%d tags=%r type=%s max_ms=%d",
        ip, len(req.lyric), tags, req.generate_type, req.max_audio_length_ms,
    )

    t0 = time.time()
    status = "ok"
    output_bytes = 0
    loop = asyncio.get_running_loop()

    async with _gpu_sem:
        try:
            audio_bytes = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _run_generation(
                        req.lyric, tags, req.generate_type,
                        req.max_audio_length_ms, req.temperature, req.topk, req.cfg_scale,
                    ),
                ),
                timeout=300,
            )
            output_bytes = len(audio_bytes)
        except asyncio.TimeoutError:
            status = "error: generation timed out"
            log.error("[%s] /generate timed out after 300s", ip)
            raise HTTPException(504, "Generation timed out after 300s")
        except Exception as exc:
            status = f"error: {exc}"
            log.exception("Generation failed")
            raise HTTPException(500, str(exc))
        finally:
            elapsed = time.time() - t0
            log_usage({
                "endpoint": "/generate",
                "client_ip": ip,
                "lyric_length": len(req.lyric),
                "tags": tags,
                "generate_type": req.generate_type,
                "max_audio_length_ms": req.max_audio_length_ms,
                "generation_time_s": round(elapsed, 2),
                "output_bytes": output_bytes,
                "status": status,
            })

    elapsed = time.time() - t0
    log.info("[%s] /generate done in %.1fs, %d bytes", ip, elapsed, output_bytes)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=heartmula_output.mp3",
            "X-Generation-Time": str(round(elapsed, 2)),
        },
    )
