"""
SongGeneration v2-large — FastAPI inference server.

Endpoints
---------
POST   /v1/music_generation  — MiniMax-compatible endpoint (Bearer auth, audio_setting)
POST   /generate             — legacy: lyrics + descriptions to song (JSON body)
POST   /generate_stream      — legacy: streaming MP3 chunks
GET    /health               — liveness check
GET    /usage                — tail recent API usage log entries

Auth
----
Set SONGGEN_API_TOKEN env var to require Bearer token on all POST endpoints.
Leave unset to disable auth (open access).
"""

import asyncio
import gc
import io
import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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

# Bearer-token auth — set SONGGEN_API_TOKEN to enable, leave unset for open access.
API_TOKEN = os.environ.get("SONGGEN_API_TOKEN", "")
_bearer = HTTPBearer(auto_error=False)

app = FastAPI(title="SongGeneration v2-large API", version="2.3.0")


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if API_TOKEN and (credentials is None or credentials.credentials != API_TOKEN):
        raise HTTPException(401, "Invalid or missing Bearer token")

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


def _run_generation_raw(
    lyric: str,
    descriptions: Optional[str],
    auto_prompt_audio_type: Optional[str],
    generate_type: str,
) -> dict:
    """Like _run_generation but returns the raw waveform tensor instead of MP3 bytes."""
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

        wav_cpu = wav_mixed[0].cpu().float()
        return {"wav_cpu": wav_cpu, "mixed_duration_s": wav_cpu.shape[-1] / _sample_rate}
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
# MiniMax-compatible models
# ---------------------------------------------------------------------------

class AudioSetting(BaseModel):
    sample_rate: int = Field(44100, description="Output sample rate in Hz (model native; resampled if different)")
    bitrate: int = Field(128000, description="Target MP3 bitrate in bps (e.g. 128000, 256000). Ignored for WAV.")
    format: str = Field("mp3", description="Output format: 'mp3' or 'wav'")


class MiniMaxRequest(BaseModel):
    model: str = Field("SongGeneration-v2-large", description="Model identifier (informational; server always runs SongGeneration-v2-large)")
    prompt: Optional[str] = Field(None, description=(
        "Style prompt, e.g. 'Indie folk, melancholic, introspective'. "
        "Maps to descriptions."
    ))
    lyrics: Optional[str] = Field(None, description=(
        "Song lyrics with [verse], [chorus], etc. labels. "
        "Omit or pass empty string for instrumental (bgm) output."
    ))
    audio_setting: Optional[AudioSetting] = Field(None, description="Output audio format settings")
    auto_prompt_audio_type: Optional[str] = Field(None, description=(
        "Auto-select a reference audio style. Options: "
        "Pop, Latin, Rock, Electronic, Metal, Country, R&B/Soul, Ballad, "
        "Jazz, World, Hip-Hop, Funk, Soundtrack, Auto."
    ))
    generate_type: Optional[str] = Field(None, description=(
        "Output type override: 'bgm', 'mixed', or 'vocal'. "
        "Default: 'mixed' when lyrics are provided, 'bgm' otherwise."
    ))


def _encode_audio(wav_cpu: torch.Tensor, native_sr: int, setting: Optional[AudioSetting]) -> tuple[bytes, str]:
    """Encode waveform to bytes according to AudioSetting. Returns (bytes, media_type)."""
    fmt = (setting.format if setting else "mp3").lower()
    target_sr = setting.sample_rate if setting else native_sr
    bitrate = setting.bitrate if setting else 128000

    if target_sr != native_sr:
        wav_cpu = torchaudio.functional.resample(wav_cpu, native_sr, target_sr)

    buf = io.BytesIO()
    if fmt == "wav":
        torchaudio.save(buf, wav_cpu, target_sr, format="wav")
        return buf.getvalue(), "audio/wav"

    torchaudio.save(buf, wav_cpu, target_sr, format="mp3")
    return buf.getvalue(), "audio/mpeg"


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
    "/v1/music_generation",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}},
    summary="MiniMax-compatible music generation endpoint",
    dependencies=[Depends(_check_auth)],
)
async def music_generation(req: MiniMaxRequest, request: Request):
    if not _ready:
        raise HTTPException(503, "Model not loaded yet.")

    # Derive generate_type from request
    has_lyrics = bool(req.lyrics and req.lyrics.strip() and req.lyrics.strip() != ".")
    if req.generate_type:
        if req.generate_type not in ("mixed", "vocal", "bgm"):
            raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}. Use 'bgm', 'mixed', or 'vocal'.")
        generate_type = req.generate_type
    else:
        generate_type = "mixed" if has_lyrics else "bgm"

    lyric = req.lyrics if has_lyrics else "."
    ip = client_ip(request)
    log.info(
        "[%s] /v1/music_generation lyric_len=%d prompt=%r auto_prompt=%r type=%s",
        ip, len(lyric), req.prompt, req.auto_prompt_audio_type, generate_type,
    )

    t0 = time.time()
    status = "ok"
    output_bytes = 0
    loop = asyncio.get_running_loop()

    if await request.is_disconnected():
        return Response(status_code=499)

    async with _gpu_sem:
        if await request.is_disconnected():
            return Response(status_code=499)
        try:
            raw_result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: _run_generation_raw(lyric, req.prompt, req.auto_prompt_audio_type, generate_type),
                ),
                timeout=300,
            )
            audio_bytes, media_type = _encode_audio(raw_result["wav_cpu"], _sample_rate, req.audio_setting)
            output_bytes = len(audio_bytes)

        except asyncio.TimeoutError:
            status = "error: generation timed out"
            raise HTTPException(504, "Generation timed out after 300s")
        except Exception as exc:
            status = f"error: {exc}"
            log.exception("Generation failed")
            raise HTTPException(500, str(exc))
        finally:
            elapsed = time.time() - t0
            log_usage({
                "endpoint": "/v1/music_generation",
                "client_ip": ip,
                "lyric_length": len(lyric),
                "prompt": req.prompt,
                "auto_prompt_audio_type": req.auto_prompt_audio_type,
                "generate_type": generate_type,
                "generation_time_s": round(elapsed, 2),
                "output_bytes": output_bytes,
                "status": status,
            })

    log.info("[%s] /v1/music_generation done in %.1fs, %d bytes", ip, elapsed, output_bytes)
    ext = "wav" if media_type == "audio/wav" else "mp3"
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=songgen_output.{ext}",
            "X-Generation-Time": str(round(elapsed, 2)),
        },
    )


@app.post(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}},
    summary="Lyrics + descriptions to song (bgm = pure instrumental)",
    dependencies=[Depends(_check_auth)],
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


# ---------------------------------------------------------------------------
# Streaming endpoint (Flavor B — pipelined LM + diffusion)
# ---------------------------------------------------------------------------

# Diffusion decoder min window = 40s of audio = 1000 tokens at 25 Hz.
# We emit one MP3 chunk per fully-decoded window. See generate_septoken.py:180.
_STREAM_WINDOW_TOKENS = 1000
# How often the LM hands us a snapshot of the in-progress sequence.
# 250 tokens ~= 10 s of audio at 25 Hz.
_STREAM_LM_CALLBACK_EVERY = 250


def _stream_token_producer(
    lyric: str,
    descriptions: Optional[str],
    auto_prompt_audio_type: Optional[str],
    generate_type: str,
    token_queue: "queue.Queue",
):
    """Run the LM in a worker thread; push token snapshots onto a queue.

    Messages on the queue:
        ("tokens", gen_seq_cpu, offset, total)  — mid-generation snapshot
        ("done",   final_tokens_cpu)            — generation finished
        ("error",  exception)                   — LM crashed
    """
    try:
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

        lyric_text = lyric.replace("  ", " ") if generate_type != "bgm" else "."

        def lm_callback(gen_seq: torch.Tensor, offset: int, total: int):
            # .cpu() forces a sync but only blocks this LM thread, not the asyncio loop.
            token_queue.put(("tokens", gen_seq.detach().cpu(), offset, total))

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                tokens = _model.generate(
                    lyrics=[lyric_text],
                    descriptions=[desc],
                    melody_wavs=pmt_wav,
                    vocal_wavs=vocal_wav,
                    bgm_wavs=bgm_wav,
                    melody_is_wav=melody_is_wav,
                    return_tokens=True,
                    token_callback=lm_callback,
                    callback_every=_STREAM_LM_CALLBACK_EVERY,
                )
        token_queue.put(("done", tokens.detach().cpu()))
    except Exception as exc:
        log.exception("LM stream worker crashed")
        token_queue.put(("error", exc))


async def _stream_chunks(
    lyric: str,
    descriptions: Optional[str],
    auto_prompt_audio_type: Optional[str],
    generate_type: str,
    ip: str,
):
    """Async generator that yields MP3 chunks as the song is decoded.

    Pipelines LM and diffusion: the LM runs in a background thread and pushes
    token snapshots; this coroutine drains them, decodes complete 40 s windows
    on the GPU, encodes each to MP3, and yields the bytes to the client.
    """
    token_queue: queue.Queue = queue.Queue()
    # Pattern object (delay layout) for the full max-length sequence.
    # lru_cache means this hits the same instance the LM thread is using.
    total_gen_len = int(_model.duration * _model.frame_rate)
    pattern = _model.lm.pattern_provider.get_pattern(total_gen_len)

    threading.Thread(
        target=_stream_token_producer,
        args=(lyric, descriptions, auto_prompt_audio_type, generate_type, token_queue),
        daemon=True,
        name="lm-stream-worker",
    ).start()

    loop = asyncio.get_running_loop()
    last_decoded = 0  # number of LM timesteps already emitted as audio
    final_tokens: Optional[torch.Tensor] = None
    t0 = time.time()
    chunks_emitted = 0
    bytes_emitted = 0

    def _decode_and_encode(codes_slice_cpu: torch.Tensor) -> bytes:
        """Run diffusion + MP3 encode on a token slice. Blocking — call from executor."""
        codes_slice = codes_slice_cpu.cuda()
        with torch.no_grad():
            wav = _model.generate_audio(codes_slice, chunked=True, gen_type=generate_type)
        wav_cpu = wav[0].cpu().float()
        buf = io.BytesIO()
        torchaudio.save(buf, wav_cpu, _sample_rate, format="mp3")
        return buf.getvalue()

    try:
        while True:
            msg = await loop.run_in_executor(None, token_queue.get)
            kind = msg[0]

            if kind == "error":
                raise msg[1]

            if kind == "done":
                final_tokens = msg[1]  # [B, K, T_actual] already pattern-reverted
                break

            # kind == "tokens"
            gen_seq_cpu = msg[1]
            out_codes, _, _ = pattern.revert_pattern_sequence(
                gen_seq_cpu, special_token=-1
            )
            # safe T = positions where ALL codebooks have a valid (>=0) token
            valid_per_t = (out_codes >= 0).all(dim=1)  # [B, T_full]
            t_safe = int(valid_per_t[0].sum().item())

            while t_safe - last_decoded >= _STREAM_WINDOW_TOKENS:
                slice_cpu = out_codes[..., last_decoded:last_decoded + _STREAM_WINDOW_TOKENS]
                mp3_bytes = await loop.run_in_executor(None, _decode_and_encode, slice_cpu)
                last_decoded += _STREAM_WINDOW_TOKENS
                chunks_emitted += 1
                bytes_emitted += len(mp3_bytes)
                log.info("[%s] /generate_stream chunk %d (%d B, t_safe=%d)",
                         ip, chunks_emitted, len(mp3_bytes), t_safe)
                yield mp3_bytes

        # LM done — emit any residual tokens (likely <1000 so Tango will pad-repeat).
        if final_tokens is not None and final_tokens.shape[-1] > last_decoded:
            slice_cpu = final_tokens[..., last_decoded:]
            mp3_bytes = await loop.run_in_executor(None, _decode_and_encode, slice_cpu)
            chunks_emitted += 1
            bytes_emitted += len(mp3_bytes)
            log.info("[%s] /generate_stream final chunk %d (%d B)",
                     ip, chunks_emitted, len(mp3_bytes))
            yield mp3_bytes

        elapsed = time.time() - t0
        log.info("[%s] /generate_stream done: %d chunks, %d B, %.1fs",
                 ip, chunks_emitted, bytes_emitted, elapsed)
        log_usage({
            "endpoint": "/generate_stream",
            "client_ip": ip,
            "lyric_length": len(lyric),
            "descriptions": descriptions,
            "auto_prompt_audio_type": auto_prompt_audio_type,
            "generate_type": generate_type,
            "generation_time_s": round(elapsed, 2),
            "output_bytes": bytes_emitted,
            "chunks": chunks_emitted,
            "status": "ok",
        })
    except Exception as exc:
        elapsed = time.time() - t0
        log.exception("[%s] /generate_stream failed", ip)
        log_usage({
            "endpoint": "/generate_stream",
            "client_ip": ip,
            "lyric_length": len(lyric),
            "descriptions": descriptions,
            "auto_prompt_audio_type": auto_prompt_audio_type,
            "generate_type": generate_type,
            "generation_time_s": round(elapsed, 2),
            "output_bytes": bytes_emitted,
            "chunks": chunks_emitted,
            "status": f"error: {exc}",
        })
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.post(
    "/generate_stream",
    summary="Streaming variant: yields MP3 chunks as the song is decoded",
    dependencies=[Depends(_check_auth)],
)
async def generate_stream(req: GenerateRequest, request: Request):
    if not _ready:
        raise HTTPException(503, "Model not loaded yet.")
    if req.generate_type not in ("mixed", "vocal", "bgm"):
        raise HTTPException(
            400,
            f"Invalid generate_type: {req.generate_type!r}. Use 'bgm', 'mixed', or 'vocal'.",
        )

    ip = client_ip(request)
    log.info(
        "[%s] /generate_stream lyric_len=%d descriptions=%r auto_prompt=%r type=%s",
        ip, len(req.lyric), req.descriptions, req.auto_prompt_audio_type, req.generate_type,
    )

    # Hold the GPU semaphore for the WHOLE streaming session — concurrent
    # generations would crush single-GPU latency far worse than serial.
    await _gpu_sem.acquire()

    async def _gen():
        try:
            async for chunk in _stream_chunks(
                req.lyric, req.descriptions, req.auto_prompt_audio_type,
                req.generate_type, ip,
            ):
                yield chunk
        finally:
            _gpu_sem.release()

    return StreamingResponse(
        _gen(),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=songgen_stream.mp3",
            "X-Stream-Mode": "chunked-mp3",
            "Cache-Control": "no-cache, no-store",
        },
    )
