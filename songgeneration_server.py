"""
SongGeneration v2-large — FastAPI inference server.

Endpoints
---------
POST   /generate  — lyrics + descriptions to song (JSON body)
GET    /health    — liveness check
GET    /usage     — tail recent API usage log entries
"""

import asyncio
import gc
import io
import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SONGGEN_ROOT = Path("/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen/songgeneration")
CKPT_PATH = SONGGEN_ROOT / "ckpt" / "songgeneration_v2_large"

_extra_paths = [
    str(SONGGEN_ROOT / "codeclm" / "tokenizer"),
    str(SONGGEN_ROOT),
    str(SONGGEN_ROOT / "codeclm" / "tokenizer" / "Flow1dVAE"),
]
for p in _extra_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TRANSFORMERS_CACHE", str(SONGGEN_ROOT / "third_party" / "hub"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.chdir(SONGGEN_ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "server.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
log = logging.getLogger(__name__)

usage_logger = logging.getLogger("usage")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False
_usage_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "api_usage.log", maxBytes=10 * 1024 * 1024, backupCount=10
)
_usage_handler.setFormatter(logging.Formatter("%(message)s"))
usage_logger.addHandler(_usage_handler)


def log_usage(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    usage_logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Model globals
# ---------------------------------------------------------------------------
MODEL_ID = "SongGeneration-v2-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SongGeneration v2-large API", version="2.2.0")

# Semaphore ensures only one generation runs at a time (serial GPU access).
_gpu_sem: asyncio.Semaphore | None = None

_cfg = None
_auto_prompt = None
_sample_rate = None
_audiolm = None
_sep_tok = None
_model = None
_ready = False

auto_prompt_types = [
    "Pop", "Latin", "Rock", "Electronic", "Metal", "Country",
    "R&B/Soul", "Ballad", "Jazz", "World", "Hip-Hop", "Funk",
    "Soundtrack", "Auto",
]


@app.on_event("startup")
async def load_model():
    global _cfg, _auto_prompt, _sample_rate, _audiolm, _sep_tok, _model, _ready, _gpu_sem

    _gpu_sem = asyncio.Semaphore(1)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(int(time.time()))

    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
    OmegaConf.register_new_resolver("get_fname", lambda: "server", replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)), replace=True)

    from codeclm.models import builders, CodecLM

    cfg_path = str(CKPT_PATH / "config.yaml")
    model_pt_path = str(CKPT_PATH / "model.pt")

    log.info("Loading %s from %s on %s\u2026", MODEL_ID, CKPT_PATH, DEVICE)
    t0 = time.time()

    _cfg = OmegaConf.load(cfg_path)
    _cfg.lm.use_flash_attn_2 = True
    _cfg.mode = "inference"
    _sample_rate = _cfg.sample_rate

    auto_prompt_path = str(SONGGEN_ROOT / "tools" / "new_auto_prompt.pt")
    _auto_prompt = torch.load(auto_prompt_path, map_location="cpu")
    log.info("Auto prompt library loaded")

    _audiolm = builders.get_lm_model(_cfg, version="v2")
    checkpoint = torch.load(model_pt_path, map_location="cpu", mmap=True)
    audiolm_state_dict = {k.replace("audiolm.", ""): v for k, v in checkpoint.items() if k.startswith("audiolm")}
    _audiolm.load_state_dict(audiolm_state_dict, strict=False)
    del checkpoint, audiolm_state_dict
    gc.collect()
    _audiolm = _audiolm.eval().cuda().to(torch.float16)
    log.info("LM loaded on GPU")

    _sep_tok = builders.get_audio_tokenizer_model_cpu(_cfg.audio_tokenizer_checkpoint_sep, _cfg)
    device = "cuda:0"
    _sep_tok.model.device = device
    _sep_tok.model.vae = _sep_tok.model.vae.to(device)
    _sep_tok.model.model.device = torch.device(device)
    _sep_tok.model.model = _sep_tok.model.model.to(device)
    _sep_tok = _sep_tok.eval()
    gc.collect()
    torch.cuda.empty_cache()
    log.info("Diffusion tokenizer loaded on GPU")

    _model = CodecLM(
        name="songgen_server",
        lm=_audiolm,
        audiotokenizer=None,
        max_duration=_cfg.max_dur,
        seperate_tokenizer=_sep_tok,
    )
    _model.set_generation_params(
        duration=_cfg.max_dur,
        extend_stride=5,
        temperature=0.8,
        cfg_coef=1.5,
        top_k=5000,
        top_p=0.0,
        record_tokens=True,
        record_window=50,
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


def _check_language(text: str) -> str:
    import re
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    if len(text) == 0:
        return "en"
    return "zh" if chinese_count / len(text) >= 0.2 else "en"


def _run_generation(
    lyric: str,
    descriptions: Optional[str],
    auto_prompt_audio_type: Optional[str],
    generate_type: str,
) -> dict:
    """Generate song. Both LM and diffusion tokenizer are resident on GPU."""
    if generate_type == "bgm":
        desc = "[Musicality-very-high], [Pure-Music], " + (descriptions.lower() if descriptions else ".")
    elif descriptions:
        desc = "[Musicality-very-high], " + descriptions.lower()
    else:
        desc = "[Musicality-very-high], ."

    if auto_prompt_audio_type and auto_prompt_audio_type in auto_prompt_types:
        lang = _check_language(lyric)
        prompts = _auto_prompt[auto_prompt_audio_type][lang]
        prompt_token = prompts[np.random.randint(0, len(prompts))]
        pmt_wav = prompt_token[:, [0], :]
        vocal_wav = prompt_token[:, [1], :]
        bgm_wav = prompt_token[:, [2], :]
        melody_is_wav = False
    else:
        pmt_wav = None
        vocal_wav = None
        bgm_wav = None
        melody_is_wav = True

    generate_inp = {
        "lyrics": [lyric.replace("  ", " ")] if generate_type != "bgm" else ".",
        "descriptions": [desc],
        "melody_wavs": pmt_wav,
        "vocal_wavs": vocal_wav,
        "bgm_wavs": bgm_wav,
        "melody_is_wav": melody_is_wav,
    }

    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = _model.generate(**generate_inp, return_tokens=True)

        with torch.no_grad():
            wav_mixed = _model.generate_audio(tokens, chunked=True, gen_type=generate_type)

        mixed_cpu = wav_mixed[0].cpu().float()
        mixed_duration_s = mixed_cpu.shape[-1] / _sample_rate

        buf = io.BytesIO()
        torchaudio.save(buf, mixed_cpu, _sample_rate, format="mp3")
        return {"mixed": buf.getvalue(), "mixed_duration_s": mixed_duration_s}
    except RuntimeError:
        gc.collect()
        torch.cuda.empty_cache()
        raise


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    lyric: str = Field(..., description=(
        "Song lyrics in SongGeneration format. Use structure labels like "
        "[verse], [chorus], [bridge], [intro-short], [outro-medium], etc. "
        "Separate sections with semicolons (;). Separate sentences with periods (.)."
    ))
    descriptions: Optional[str] = Field(None, description=(
        "Comma-separated style tags, e.g. 'electronic, energetic, synthesizer'. "
        "Omit to let the model decide."
    ))
    auto_prompt_audio_type: Optional[str] = Field(None, description=(
        "Auto-select a reference audio style. Options: "
        "Pop, Latin, Rock, Electronic, Metal, Country, R&B/Soul, Ballad, "
        "Jazz, World, Hip-Hop, Funk, Soundtrack, Auto."
    ))
    generate_type: str = Field("bgm", description=(
        "Output type: 'bgm' (instrumental only, default), 'mixed' (vocals+accompaniment), "
        "'vocal' (vocals only)."
    ))


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
    usage_log = LOG_DIR / "api_usage.log"
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
    summary="Lyrics + descriptions to song (bgm = pure instrumental)",
)
async def generate(req: GenerateRequest, request: Request):
    if not _ready:
        raise HTTPException(503, "Model not loaded yet.")
    if req.generate_type not in ("mixed", "vocal", "bgm"):
        raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}. Use 'bgm', 'mixed', or 'vocal'.")

    ip = client_ip(request)
    log.info(
        "[%s] /generate lyric_len=%d descriptions=%r auto_prompt=%r type=%s",
        ip, len(req.lyric), req.descriptions, req.auto_prompt_audio_type, req.generate_type,
    )

    t0 = time.time()
    status = "ok"
    output_bytes = 0
    loop = asyncio.get_running_loop()

    # Skip GPU work if the client already went away (e.g. user pressed Stop).
    if await request.is_disconnected():
        log.info("[%s] /generate -- client disconnected before queueing, skipping", ip)
        return Response(status_code=499)

    async with _gpu_sem:
        # Client may have disconnected while waiting for a prior generation.
        if await request.is_disconnected():
            log.info("[%s] /generate -- client disconnected while waiting for GPU sem, skipping", ip)
            return Response(status_code=499)
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _run_generation(
                        req.lyric, req.descriptions, req.auto_prompt_audio_type,
                        req.generate_type,
                    ),
                ),
                timeout=300,
            )
            audio_bytes = result["mixed"]
            output_bytes = len(audio_bytes)

        except asyncio.CancelledError:
            # Uvicorn detected client disconnect mid-inference. The inference thread
            # cannot be interrupted (Python limitation) and will complete in the
            # background, but the semaphore is released so the next request proceeds.
            status = "cancelled: client disconnected mid-inference"
            log.info("[%s] /generate cancelled mid-inference (client disconnected)", ip)
            raise
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
                "descriptions": req.descriptions,
                "auto_prompt_audio_type": req.auto_prompt_audio_type,
                "generate_type": req.generate_type,
                "generation_time_s": round(elapsed, 2),
                "output_bytes": output_bytes,
                "status": status,
            })

    log.info("[%s] /generate done in %.1fs, %d bytes", ip, elapsed, output_bytes)
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=songgen_output.mp3",
            "X-Generation-Time": str(round(elapsed, 2)),
        },
    )
