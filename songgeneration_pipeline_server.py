"""
SongGeneration v2-large — Pipeline inference server.

Architecture
------------
- LM workers    : Run on auto-detected or explicit GPU ids. Load full model. Run batched
                  autoregressive generation with a token_callback that pushes DiffusionJobs
                  onto a shared queue as soon as 1000 tokens (40 s) are ready per request.
- Diffusion workers : Run on remaining GPUs. Load full model (only use sep_tok).
                  Pull DiffusionJobs, decode audio windows, encode MP3, push ChunkResults.
- Main process  : FastAPI, batch scheduler, result listener that routes MP3 chunks to
                  per-request asyncio queues for streaming HTTP responses.

GPU auto-split (when env vars not set)
--------------------------------------
  N=1  → lm=[0]           diff=[0]   (shared)
  N=2  → lm=[0]           diff=[1]
  N=4  → lm=[0,1,2]       diff=[3]
  N=8  → lm=[0..6]        diff=[7]
  Rule: diff_count = max(1, N // 6)

Endpoints
---------
POST /v1/music_generation — MiniMax-compatible endpoint (Bearer auth, audio_setting)
POST /generate            — blocking, returns full MP3 (backward compat)
POST /generate_stream     — chunked StreamingResponse; first audio ~40 s after request
GET  /health              — liveness + worker stats
GET  /usage               — tail recent usage log

Auth
----
Set SONGGEN_API_TOKEN env var to require Bearer token on all POST endpoints.
Leave unset to disable auth (open access).

Environment variables
---------------------
SONGGEN_LM_GPU_IDS      Comma-separated GPU ids for LM workers   (auto if unset)
SONGGEN_DIFF_GPU_IDS    Comma-separated GPU ids for Diff workers  (auto if unset)
SONGGEN_BATCH_MAX       Max requests per LM batch                 (default: 2)
SONGGEN_BATCH_WAIT_MS   Max ms to wait for batch to fill          (default: 500)
SONGGEN_COMPILE         Set to 1 to torch.compile the LM         (default: off)
SONGGEN_API_TOKEN       Bearer token for API auth                 (default: unset = open)
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import logging.handlers
import os
import queue as _queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Must be set BEFORE torch is imported — avoids fragmentation when LM+Diff
# share a 24 GB MIG slice.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _auto_split_gpus() -> tuple[list[int], list[int]]:
    """Return (lm_gpu_ids, diff_gpu_ids) based on available GPUs.

    diff_count = max(1, N//6). 1 diff worker handles up to ~16 LM workers
    at ~1.8 s/chunk decode vs ~180 s LM time, so the rest go to LM.
    """
    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("No CUDA GPUs found")
    diff_count = max(1, n // 6)
    lm_count   = n - diff_count
    if lm_count == 0:
        return [0], [0]
    all_ids = list(range(n))
    return all_ids[:lm_count], all_ids[lm_count:]


def _parse_gpu_ids(env_var: str) -> list[int] | None:
    val = os.environ.get(env_var, "").strip()
    if not val:
        return None
    return [int(x) for x in val.split(",")]


MAX_BATCH_SIZE = int(os.environ.get("SONGGEN_BATCH_MAX",      "2"))
BATCH_WAIT_MS  = int(os.environ.get("SONGGEN_BATCH_WAIT_MS",  "500"))
# When LM and Diff share a GPU (MIG 24 GB), streaming is counterproductive —
# the LM's KV cache (~8 GB) plus active generation activations leave no room
# for the diff worker. Disable the token_callback in shared mode: LM finishes
# fully, releases KV cache, then diff decodes all chunks sequentially.
SHARED_GPU_MODE = False  # set in lm_worker_fn based on LM_GPU_IDS vs DIFF_GPU_IDS

_lm_override   = _parse_gpu_ids("SONGGEN_LM_GPU_IDS")
_diff_override = _parse_gpu_ids("SONGGEN_DIFF_GPU_IDS")

if _lm_override is not None and _diff_override is not None:
    LM_GPU_IDS, DIFF_GPU_IDS = _lm_override, _diff_override
elif _lm_override is not None or _diff_override is not None:
    raise RuntimeError(
        "Set both SONGGEN_LM_GPU_IDS and SONGGEN_DIFF_GPU_IDS or neither."
    )
else:
    LM_GPU_IDS, DIFF_GPU_IDS = _auto_split_gpus()

# Diffusion decoder minimum window (40 s @ 25 Hz token rate)
WINDOW_TOKENS  = 1000
# How often the LM callback fires (250 tokens ≈ 10 s of audio)
CALLBACK_EVERY = 250
# RVQ audio codebook upper bound (matches rvq_bestrq_emb.codebook_size-1 = 16383).
# Any LM-emitted code > AUDIO_CODE_MAX is a structure/EOS token that the
# diffusion decoder cannot represent; decoding such codes (even via the
# defensive clamp in model_septoken.py) produces noise. We truncate the
# token stream at the first such timestep before dispatching diff jobs.
AUDIO_CODE_MAX = 16383


def _valid_prefix_len(codes_kt: torch.Tensor) -> int:
    """Length of the contiguous prefix where all codebooks are in [0, AUDIO_CODE_MAX]. Accepts [K, T]."""
    valid_t = ((codes_kt >= 0) & (codes_kt <= AUDIO_CODE_MAX)).all(dim=0)
    inv = (~valid_t).nonzero(as_tuple=True)[0]
    return int(inv[0].item()) if inv.numel() > 0 else int(valid_t.numel())

SONGGEN_ROOT = Path("/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen/songgeneration")
CKPT_PATH    = SONGGEN_ROOT / "ckpt" / "songgeneration_v2_large"
MODEL_ID     = "SongGeneration-v2-large"

auto_prompt_types = [
    "Pop", "Latin", "Rock", "Electronic", "Metal", "Country",
    "R&B/Soul", "Ballad", "Jazz", "World", "Hip-Hop", "Funk",
    "Soundtrack", "Auto",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [main] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "pipeline_server.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
log = logging.getLogger(__name__)

usage_logger = logging.getLogger("usage")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False
_uh = logging.handlers.RotatingFileHandler(
    LOG_DIR / "api_usage.log", maxBytes=10 * 1024 * 1024, backupCount=10
)
_uh.setFormatter(logging.Formatter("%(message)s"))
usage_logger.addHandler(_uh)


def log_usage(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    usage_logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Shared inter-process message types
# ---------------------------------------------------------------------------

# Sent by LM workers → diff_queue → Diffusion workers
@dataclass
class DiffusionJob:
    request_id: str
    chunk_index: int   # 0-based, for ordering at the client
    token_slice: bytes # pickled CPU tensor [1, K, T]
    gen_type:    str
    is_final:    bool  # True for the last window of this request


# Sent by Diffusion workers → output_queue → main process
@dataclass
class ChunkResult:
    request_id:  str
    chunk_index: int
    mp3_bytes:   bytes
    is_final:    bool
    error:       Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers shared by worker processes
# ---------------------------------------------------------------------------

def _setup_paths():
    extra = [
        str(SONGGEN_ROOT / "codeclm" / "tokenizer"),
        str(SONGGEN_ROOT),
        str(SONGGEN_ROOT / "codeclm" / "tokenizer" / "Flow1dVAE"),
    ]
    for p in extra:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(SONGGEN_ROOT / "third_party" / "hub"))
    os.chdir(SONGGEN_ROOT)


def _check_language(text: str) -> str:
    import re
    c = len(re.findall(r"[\u4e00-\u9fff]", text))
    return "zh" if len(text) > 0 and c / len(text) >= 0.2 else "en"


def _load_full_model(gpu_id: int, logger, role: str = "both"):
    """Load model components on cuda:{gpu_id}. Returns (model, auto_prompt, sample_rate).

    role: "lm"   — only LM weights on GPU; audio tokenizer stays on CPU (used for audio decode).
          "diff" — only audio tokenizer on GPU; LM weights stay on CPU.
          "both" — both on GPU (single-process fallback).
    Keeping unused components off the GPU lets LM + Diff share a 24 GB MIG slice.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # expandable_segments avoids fragmentation when LM+Diff share a 24 GB MIG slice
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _setup_paths()

    torch.backends.cudnn.enabled             = True
    torch.backends.cudnn.benchmark           = True
    torch.backends.cuda.matmul.allow_tf32    = True
    torch.backends.cudnn.allow_tf32          = True
    torch.set_num_threads(1)
    np.random.seed(int.from_bytes(os.urandom(4), "big"))

    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval",      lambda x: eval(x),                              replace=True)
    OmegaConf.register_new_resolver("concat",    lambda *x: [i for s in x for i in s],          replace=True)
    OmegaConf.register_new_resolver("get_fname", lambda: "server",                               replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)),             replace=True)

    from codeclm.models import builders, CodecLM

    t0 = time.time()
    cfg = OmegaConf.load(str(CKPT_PATH / "config.yaml"))
    cfg.lm.use_flash_attn_2 = True
    cfg.mode = "inference"
    sample_rate = cfg.sample_rate

    auto_prompt = torch.load(
        str(SONGGEN_ROOT / "tools" / "new_auto_prompt.pt"), map_location="cpu"
    )

    audiolm = builders.get_lm_model(cfg, version="v2")
    ckpt = torch.load(str(CKPT_PATH / "model.pt"), map_location="cpu", mmap=True)
    sd = {k.replace("audiolm.", ""): v for k, v in ckpt.items() if k.startswith("audiolm")}
    audiolm.load_state_dict(sd, strict=False)
    del ckpt, sd
    gc.collect()
    if role in ("lm", "both"):
        audiolm = audiolm.eval().cuda().to(torch.float16)
    else:
        audiolm = audiolm.eval()   # stays on CPU; diff worker only needs lm.code_depth

    sep_tok = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    if role in ("diff", "both"):
        dev = "cuda:0"   # always 0 inside this process (CUDA_VISIBLE_DEVICES remaps it)
        sep_tok.model.device = dev
        sep_tok.model.vae    = sep_tok.model.vae.to(dev)
        sep_tok.model.model.device = torch.device(dev)
        sep_tok.model.model  = sep_tok.model.model.to(dev)
    sep_tok = sep_tok.eval()
    gc.collect()
    torch.cuda.empty_cache()

    model = CodecLM(
        name=f"pipeline_gpu{gpu_id}",
        lm=audiolm,
        audiotokenizer=None,
        max_duration=cfg.max_dur,
        seperate_tokenizer=sep_tok,
    )
    model.set_generation_params(
        duration=cfg.max_dur,
        extend_stride=5,
        temperature=0.8,
        cfg_coef=1.5,
        top_k=5000,
        top_p=0.0,
        record_tokens=True,
        record_window=50,
    )

    if role == "lm" and os.environ.get("SONGGEN_COMPILE", "").strip() == "1":
        import torch._dynamo as _dynamo
        _dynamo.config.suppress_errors = True  # fall back to eager on unsupported ops
        model.lm = torch.compile(model.lm, mode="reduce-overhead", dynamic=False)
        logger.info("LM compiled with torch.compile (reduce-overhead, dynamic) on GPU %d", gpu_id)

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model loaded (role=%s) on physical GPU %d in %.1fs",
                role, gpu_id, time.time() - t0)
    return model, auto_prompt, sample_rate


# ---------------------------------------------------------------------------
# LM Worker
# ---------------------------------------------------------------------------

def lm_worker_fn(
    worker_id:    int,
    gpu_id:       int,
    lm_in_queue:  mp.Queue,   # receives LM batch jobs
    diff_queue:   mp.Queue,   # shared with all diff workers
    out_queue:    mp.Queue,   # results back to main
):
    global SHARED_GPU_MODE
    SHARED_GPU_MODE = bool(set(LM_GPU_IDS) & set(DIFF_GPU_IDS))

    log = logging.getLogger(f"lm-worker-{gpu_id}")
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s [lm-gpu{gpu_id}] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

    model, auto_prompt, sample_rate = _load_full_model(gpu_id, log, role="lm")
    log.info("LM worker mode: %s", "shared-GPU (non-streaming)" if SHARED_GPU_MODE else "dedicated (streaming)")

    total_gen_len = int(model.duration * model.frame_rate)
    pattern = model.lm.pattern_provider.get_pattern(total_gen_len)

    out_queue.put({"type": "ready", "role": "lm", "worker_id": worker_id, "gpu_id": gpu_id})
    log.info("LM worker ready (gpu_id=%d)", gpu_id)

    while True:
        msg = lm_in_queue.get()
        if msg is None:
            log.info("Shutdown")
            break

        batch_id = msg["batch_id"]
        requests = msg["requests"]
        B = len(requests)
        log.info("Batch %s: %d request(s)", batch_id, B)
        t0 = time.time()

        # Per-request streaming state
        last_decoded  = [0] * B   # number of LM timesteps already sent to diffusion
        chunk_indices = [0] * B   # next chunk index per request

        # Build description strings and prompt tensors
        all_lyrics, all_descs = [], []
        all_pmt, all_vocal, all_bgm = [], [], []
        any_auto = False

        for req in requests:
            gt = req.get("generate_type", "bgm")
            desc = req.get("descriptions") or ""
            if gt == "bgm":
                d = "[Musicality-very-high], [Pure-Music], " + (desc.lower() if desc else ".")
            elif desc:
                d = "[Musicality-very-high], " + desc.lower()
            else:
                d = "[Musicality-very-high], ."

            lyric = req.get("lyric", ".").replace("  ", " ") if gt != "bgm" else "."

            apt = req.get("auto_prompt_audio_type")
            if apt and apt in auto_prompt_types:
                lang = _check_language(lyric)
                prompts = auto_prompt[apt][lang]
                pt = prompts[np.random.randint(0, len(prompts))]
                all_pmt.append(pt[:, [0], :])
                all_vocal.append(pt[:, [1], :])
                all_bgm.append(pt[:, [2], :])
                any_auto = True
            else:
                all_pmt.append(None)
                all_vocal.append(None)
                all_bgm.append(None)

            all_lyrics.append(lyric)
            all_descs.append(d)

        if any_auto:
            ref = next(p for p in all_pmt if p is not None)
            rs  = ref.shape
            for i in range(B):
                if all_pmt[i] is None:
                    all_pmt[i]   = torch.full(rs, 16385, dtype=ref.dtype)
                    all_vocal[i] = torch.full(rs, 16385, dtype=ref.dtype)
                    all_bgm[i]   = torch.full(rs, 16385, dtype=ref.dtype)
            stacked_pmt   = torch.cat(all_pmt,   dim=0)
            stacked_vocal = torch.cat(all_vocal, dim=0)
            stacked_bgm   = torch.cat(all_bgm,   dim=0)
            melody_is_wav = False
        else:
            stacked_pmt = stacked_vocal = stacked_bgm = None
            melody_is_wav = True

        def token_callback(gen_seq: torch.Tensor, offset: int, total: int):
            """Called every CALLBACK_EVERY LM steps with shape [B, K, S]."""
            gen_seq_cpu = gen_seq.detach().cpu()
            out_codes, _, _ = pattern.revert_pattern_sequence(gen_seq_cpu, special_token=-1)

            for i in range(B):
                # Contiguous prefix of timesteps where every codebook is a
                # valid audio code (0..AUDIO_CODE_MAX). Stops at the first
                # structure/EOS token to avoid streaming garbage to diff.
                t_safe = _valid_prefix_len(out_codes[i])
                while t_safe - last_decoded[i] >= WINDOW_TOKENS:
                    sl = out_codes[i:i+1, :, last_decoded[i]:last_decoded[i] + WINDOW_TOKENS]
                    diff_queue.put(DiffusionJob(
                        request_id=requests[i]["request_id"],
                        chunk_index=chunk_indices[i],
                        token_slice=sl.numpy().tobytes(),   # pickle-free bytes
                        gen_type=requests[i].get("generate_type", "bgm"),
                        is_final=False,
                    ))
                    chunk_indices[i] += 1
                    last_decoded[i]  += WINDOW_TOKENS

        # Streaming disabled when LM+Diff share a GPU (tight 24 GB MIG slice).
        # In that case the LM's KV cache during generation leaves no room for
        # concurrent diff decode; we release it after generate() instead.
        effective_cb = None if SHARED_GPU_MODE else token_callback

        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    tokens = model.generate(
                        lyrics=all_lyrics,
                        descriptions=all_descs,
                        melody_wavs=stacked_pmt,
                        vocal_wavs=stacked_vocal,
                        bgm_wavs=stacked_bgm,
                        melody_is_wav=melody_is_wav,
                        return_tokens=True,
                        token_callback=effective_cb,
                        callback_every=CALLBACK_EVERY,
                    )

            # Release the LM's KV cache before the diff worker starts — this
            # frees ~8 GB on the shared GPU so diff has room to decode.
            try:
                model.lm.reset_streaming()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

            # In shared-GPU mode, all chunks must be queued now (no streaming
            # callback fired during generation). Split tokens into WINDOW_TOKENS
            # chunks plus a residual.
            if SHARED_GPU_MODE:
                for i in range(B):
                    t_eff = _valid_prefix_len(tokens[i])
                    if t_eff == 0:
                        continue  # error path below handles "no chunks queued"
                    for start in range(0, t_eff, WINDOW_TOKENS):
                        end   = min(start + WINDOW_TOKENS, t_eff)
                        sl    = tokens[i:i+1, :, start:end]
                        diff_queue.put(DiffusionJob(
                            request_id=requests[i]["request_id"],
                            chunk_index=chunk_indices[i],
                            token_slice=sl.cpu().numpy().tobytes(),
                            gen_type=requests[i].get("generate_type", "bgm"),
                            is_final=(end == t_eff),
                        ))
                        chunk_indices[i] += 1
                        last_decoded[i]  = end

            # Final residual: tokens beyond what callback already dispatched
            # (skipped in SHARED_GPU_MODE — everything was already queued above)
            chunks_queued = {}
            for i in range(B):
                rid = requests[i]["request_id"]
                if SHARED_GPU_MODE:
                    if chunk_indices[i] == 0:
                        out_queue.put({"type": "chunk_error",
                                       "request_id": rid,
                                       "error": "LM produced no tokens"})
                elif _valid_prefix_len(tokens[i]) > last_decoded[i]:
                    # Only stream the valid prefix of the residual; the rest
                    # is structure/EOS tokens that would decode as noise.
                    t_eff = _valid_prefix_len(tokens[i])
                    sl = tokens[i:i+1, :, last_decoded[i]:t_eff]
                    diff_queue.put(DiffusionJob(
                        request_id=rid,
                        chunk_index=chunk_indices[i],
                        token_slice=sl.cpu().numpy().tobytes(),
                        gen_type=requests[i].get("generate_type", "bgm"),
                        is_final=True,
                    ))
                    chunk_indices[i] += 1
                elif chunk_indices[i] > 0:
                    # No residual but we already queued chunks — mark last one final.
                    # Can't retroactively update it; instead send a zero-byte sentinel.
                    diff_queue.put(DiffusionJob(
                        request_id=rid,
                        chunk_index=chunk_indices[i],
                        token_slice=b"",   # sentinel: no audio
                        gen_type=requests[i].get("generate_type", "bgm"),
                        is_final=True,
                    ))
                    chunk_indices[i] += 1
                else:
                    # Nothing generated at all — error path
                    out_queue.put({"type": "chunk_error",
                                   "request_id": rid,
                                   "error": "LM produced no tokens"})
                chunks_queued[rid] = chunk_indices[i]

            out_queue.put({
                "type": "lm_batch_done",
                "batch_id": batch_id,
                "chunks_queued": chunks_queued,
                "elapsed": time.time() - t0,
            })
            log.info("Batch %s: LM done in %.1fs, %s chunks queued",
                     batch_id, time.time() - t0,
                     {r: n for r, n in chunks_queued.items()})

            del tokens, stacked_pmt, stacked_vocal, stacked_bgm
            # Release the Llama KV cache held in self._streaming_state —
            # otherwise it keeps ~8 GB on the GPU after generate() returns.
            try:
                model.lm.reset_streaming()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as exc:
            log.exception("Batch %s LM failed: %s", batch_id, exc)
            try:
                model.lm.reset_streaming()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            for req in requests:
                out_queue.put({"type": "chunk_error",
                               "request_id": req["request_id"],
                               "error": str(exc)})
            out_queue.put({"type": "lm_batch_done", "batch_id": batch_id,
                           "chunks_queued": {}, "elapsed": time.time() - t0})


# ---------------------------------------------------------------------------
# Diffusion Worker
# ---------------------------------------------------------------------------

def diff_worker_fn(
    worker_id:   int,
    gpu_id:      int,
    diff_queue:  mp.Queue,
    out_queue:   mp.Queue,
):
    log = logging.getLogger(f"diff-worker-{gpu_id}")
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s [diff-gpu{gpu_id}] %(message)s"))
    log.addHandler(h)
    log.setLevel(logging.INFO)

    model, _, sample_rate = _load_full_model(gpu_id, log, role="diff")

    out_queue.put({"type": "ready", "role": "diff", "worker_id": worker_id, "gpu_id": gpu_id})
    log.info("Diffusion worker ready (gpu_id=%d)", gpu_id)

    K = model.lm.code_depth   # number of codebooks

    while True:
        job = diff_queue.get()
        if job is None:
            log.info("Shutdown")
            break

        if not isinstance(job, DiffusionJob):
            continue

        t0 = time.time()

        # Zero-byte sentinel → no audio, just close the stream
        if not job.token_slice:
            out_queue.put(ChunkResult(
                request_id=job.request_id,
                chunk_index=job.chunk_index,
                mp3_bytes=b"",
                is_final=True,
            ))
            continue

        try:
            # Deserialise token slice: bytes → numpy → torch
            import numpy as _np
            arr = _np.frombuffer(job.token_slice, dtype=_np.int64).reshape(1, K, -1)
            codes = torch.from_numpy(arr.copy()).cuda()

            with torch.no_grad():
                wav = model.generate_audio(codes, chunked=True, gen_type=job.gen_type)
            wav_cpu = wav[0].cpu().float()

            buf = io.BytesIO()
            torchaudio.save(buf, wav_cpu, sample_rate, format="mp3")
            mp3 = buf.getvalue()

            out_queue.put(ChunkResult(
                request_id=job.request_id,
                chunk_index=job.chunk_index,
                mp3_bytes=mp3,
                is_final=job.is_final,
            ))
            log.info("req=%s chunk=%d %.1fs audio, %.1fs decode, final=%s",
                     job.request_id, job.chunk_index,
                     wav_cpu.shape[-1] / sample_rate, time.time() - t0, job.is_final)

            # Release intermediate tensors so the MIG slice cache doesn't grow
            del codes, wav, wav_cpu, buf, arr
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as exc:
            log.exception("Diffusion failed for req=%s chunk=%d", job.request_id, job.chunk_index)
            is_cuda_fatal = isinstance(exc, RuntimeError) and (
                "CUDA error" in str(exc) or "device-side assert" in str(exc)
            )
            gc.collect()
            if not is_cuda_fatal:
                # Only call empty_cache on a healthy CUDA context — calling it on a
                # poisoned context raises another RuntimeError and masks the real error.
                torch.cuda.empty_cache()
            out_queue.put(ChunkResult(
                request_id=job.request_id,
                chunk_index=job.chunk_index,
                mp3_bytes=b"",
                is_final=job.is_final,
                error=str(exc),
            ))
            if is_cuda_fatal:
                # CUDA device-side assert poisons the entire GPU context for this process.
                # No further tensor operations will succeed. Exit so the scheduler's
                # watchdog can restart this worker with a fresh CUDA context.
                log.error("CUDA context poisoned on gpu%d — exiting diff worker for restart",
                          gpu_id)
                break


# ---------------------------------------------------------------------------
# Pipeline Scheduler (main process)
# ---------------------------------------------------------------------------

@dataclass
class PendingRequest:
    request_id:  str
    params:      dict
    chunk_queue: asyncio.Queue   # MP3 chunks pushed here by result listener
    submit_time: float
    client_ip:   str
    total_chunks_expected: int = -1   # set when lm_batch_done received
    chunks_received:       int = 0
    # Reorder buffer: diff workers pull from a shared queue and can finish
    # out of chunk_index order. Hold chunks until the contiguous prefix from
    # next_chunk_to_emit is ready, then flush to the client queue in order.
    next_chunk_to_emit:    int = 0
    reorder_buffer:        dict = field(default_factory=dict)


class PipelineScheduler:
    def __init__(self):
        self.lm_input_queues:  list[mp.Queue] = []
        self.diff_queue:       mp.Queue       = mp.Queue()
        self.output_queue:     mp.Queue       = mp.Queue()
        self.lm_processes:    list[mp.Process] = []
        self.diff_processes:  list[mp.Process] = []

        self.available_lm: asyncio.Queue[int] = None   # index into lm_input_queues
        self.request_queue: asyncio.Queue[PendingRequest] = None

        self.pending: dict[str, PendingRequest] = {}   # request_id → PendingRequest
        self._active_lm_batches: dict[str, int] = {}  # batch_id → lm_idx (for slot release)

        self.total_batches              = 0
        self.total_requests             = 0
        self.total_abandoned            = 0  # skipped pre-batch (client gone before dispatch)
        self.total_dropped_at_dispatch  = 0  # filtered inside _dispatch_lm
        self._started                   = False

    async def start_workers_parallel(self):
        """Start all workers simultaneously and wait for all to report ready.

        With 480 GB CPU RAM available, peak load (8 workers × 16 GB = 128 GB) is fine.
        Cuts startup from ~12 min (sequential) down to ~2 min.
        """
        import queue as _q

        all_configs = (
            [(idx, gpu_id, "lm") for idx, gpu_id in enumerate(LM_GPU_IDS)]
            + [(len(LM_GPU_IDS) + idx, gpu_id, "diff") for idx, gpu_id in enumerate(DIFF_GPU_IDS)]
        )
        all_procs: list[mp.Process] = []

        # Allocate LM input queues up front so worker_id → queue index is stable
        for idx, _ in enumerate(LM_GPU_IDS):
            self.lm_input_queues.append(mp.Queue())

        # Start all workers at once
        for worker_id, gpu_id, role in all_configs:
            if role == "lm":
                p = mp.Process(
                    target=lm_worker_fn,
                    args=(worker_id, gpu_id, self.lm_input_queues[worker_id],
                          self.diff_queue, self.output_queue),
                    daemon=True, name=f"lm-gpu{gpu_id}",
                )
                p.start()
                self.lm_processes.append(p)
                log.info("Started LM worker gpu%d (pid %d)", gpu_id, p.pid)
            else:
                p = mp.Process(
                    target=diff_worker_fn,
                    args=(worker_id, gpu_id, self.diff_queue, self.output_queue),
                    daemon=True, name=f"diff-gpu{gpu_id}",
                )
                p.start()
                self.diff_processes.append(p)
                log.info("Started Diff worker gpu%d (pid %d)", gpu_id, p.pid)
            all_procs.append(p)

        # Wait for all workers to report ready
        n_total = len(all_procs)
        n_ready = 0
        deadline = time.time() + 600
        loop = asyncio.get_running_loop()

        def _drain_until_all_ready():
            nonlocal n_ready
            while n_ready < n_total:
                remaining = deadline - time.time()
                if remaining <= 0:
                    names = [p.name for p in all_procs if p.is_alive()]
                    raise TimeoutError(f"Workers did not become ready in 600s. Still alive: {names}")
                for p in all_procs:
                    if not p.is_alive() and p.exitcode != 0:
                        raise RuntimeError(f"Worker {p.name} (pid {p.pid}) died (exit {p.exitcode})")
                try:
                    msg = self.output_queue.get(timeout=min(5, remaining))
                    if isinstance(msg, dict) and msg.get("type") == "ready":
                        role = msg["role"]
                        wid  = msg["worker_id"]
                        gid  = msg["gpu_id"]
                        if role == "lm":
                            self.available_lm.put_nowait(wid)
                            log.info("LM worker %d ready (gpu %d)", wid, gid)
                        else:
                            log.info("Diff worker %d ready (gpu %d)", wid, gid)
                        n_ready += 1
                except _q.Empty:
                    continue

        await loop.run_in_executor(None, _drain_until_all_ready)
        log.info("All %d workers ready", n_total)

        self._started = True
        log.info("All workers ready (lm_workers=%d, diff_workers=%d)",
                 len(self.lm_input_queues), len(self.diff_processes))

    async def submit(self, params: dict, client_ip: str) -> tuple[str, asyncio.Queue]:
        """Submit a streaming request. Returns (request_id, chunk_queue).
        Caller must pop scheduler.pending[request_id] when done (or on
        client disconnect) to free the reorder buffer.
        Sentinel on chunk_queue: None → stream finished. Exception → error."""
        rid = uuid.uuid4().hex[:12]
        params["request_id"] = rid
        cq: asyncio.Queue = asyncio.Queue()
        pr = PendingRequest(
            request_id=rid, params=params, chunk_queue=cq,
            submit_time=time.time(), client_ip=client_ip,
        )
        self.pending[rid] = pr
        await self.request_queue.put(pr)
        return rid, cq

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

    async def result_listener(self):
        """Route output_queue messages to per-request chunk queues."""
        loop = asyncio.get_running_loop()

        def _get():
            import queue
            try:
                return self.output_queue.get(timeout=5)
            except queue.Empty:
                return None

        while True:
            msg = await loop.run_in_executor(None, _get)
            if msg is None:
                continue

            if isinstance(msg, ChunkResult):
                pr = self.pending.get(msg.request_id)
                if pr is None:
                    # Client disconnected or request already closed. Drop the chunk.
                    continue
                if msg.error:
                    # Errors force immediate close; buffered chunks are discarded.
                    await pr.chunk_queue.put(RuntimeError(msg.error))
                    await pr.chunk_queue.put(None)
                    self.pending.pop(msg.request_id, None)
                    continue

                # Buffer by chunk_index. Diff workers pull from a shared queue
                # so chunk N+1 can finish before chunk N — flush only the
                # contiguous prefix starting at next_chunk_to_emit.
                pr.reorder_buffer[msg.chunk_index] = msg
                closed = False
                while pr.next_chunk_to_emit in pr.reorder_buffer:
                    emit = pr.reorder_buffer.pop(pr.next_chunk_to_emit)
                    if emit.mp3_bytes:
                        await pr.chunk_queue.put(emit.mp3_bytes)
                    pr.chunks_received     += 1
                    pr.next_chunk_to_emit  += 1
                    if emit.is_final or (
                        pr.total_chunks_expected >= 0
                        and pr.chunks_received >= pr.total_chunks_expected
                    ):
                        await pr.chunk_queue.put(None)
                        self.pending.pop(msg.request_id, None)
                        closed = True
                        break
                if closed:
                    continue

            elif isinstance(msg, dict):
                if msg["type"] == "lm_batch_done":
                    batch_id = msg["batch_id"]
                    for rid, n_chunks in msg.get("chunks_queued", {}).items():
                        pr = self.pending.get(rid)
                        if pr:
                            pr.total_chunks_expected = n_chunks
                            # If all chunks already received, close now
                            if pr.chunks_received >= n_chunks:
                                await pr.chunk_queue.put(None)
                                self.pending.pop(rid, None)
                    lm_idx = self._active_lm_batches.pop(batch_id, None)
                    if lm_idx is not None:
                        self.available_lm.put_nowait(lm_idx)
                        log.info("Released LM worker slot %d", lm_idx)

                elif msg["type"] == "chunk_error":
                    rid = msg["request_id"]
                    pr = self.pending.get(rid)
                    if pr:
                        await pr.chunk_queue.put(RuntimeError(msg["error"]))
                        await pr.chunk_queue.put(None)
                        self.pending.pop(rid, None)

    async def watchdog_loop(self):
        """Restart diff workers that exited due to CUDA errors.

        A device-side assert poisons the GPU context for the whole process; the
        worker exits cleanly so this watchdog can spawn a fresh replacement.
        LM workers are not restarted automatically — an LM crash usually means
        the model weights need reloading (expensive); alert via logs instead.
        """
        while True:
            await asyncio.sleep(20)
            for idx, p in enumerate(self.diff_processes):
                if not p.is_alive():
                    gpu_id    = DIFF_GPU_IDS[idx]
                    worker_id = len(LM_GPU_IDS) + idx
                    log.warning(
                        "Diff worker %d (gpu%d, pid=%d, exit=%s) died — restarting",
                        idx, gpu_id, p.pid, p.exitcode,
                    )
                    new_p = mp.Process(
                        target=diff_worker_fn,
                        args=(worker_id, gpu_id, self.diff_queue, self.output_queue),
                        daemon=True, name=f"diff-gpu{gpu_id}",
                    )
                    new_p.start()
                    self.diff_processes[idx] = new_p
                    log.info("Restarted diff worker gpu%d (pid=%d) — loading model (~2 min)",
                             gpu_id, new_p.pid)
            lm_alive = sum(1 for p in self.lm_processes if p.is_alive())
            if lm_alive == 0:
                log.critical("ALL LM workers dead — server is stalled. Restart the service.")

    def shutdown(self):
        for q in self.lm_input_queues:
            q.put(None)   # rank-0 of each LM worker reads this; TP rank-0 forwards to rank-1
        for _ in self.diff_processes:
            self.diff_queue.put(None)
        all_procs = self.lm_processes + self.diff_processes
        for p in all_procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()


# ---------------------------------------------------------------------------
# FastAPI app + auth
# ---------------------------------------------------------------------------
API_TOKEN = os.environ.get("SONGGEN_API_TOKEN", "")
_bearer   = HTTPBearer(auto_error=False)

app = FastAPI(title="SongGeneration Pipeline API", version="4.1.0")
scheduler = PipelineScheduler()


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if API_TOKEN and (credentials is None or credentials.credentials != API_TOKEN):
        raise HTTPException(401, "Invalid or missing Bearer token")


async def _warmup():
    """Submit one dummy BGM generation after workers come up so the first
    real request doesn't pay the JIT cost (flash-attn kernels, cuDNN
    autotune, diffusion warmup) on top of the ~250 s cold start.

    Set SONGGEN_SKIP_WARMUP=1 to disable (e.g. during dev restarts).
    """
    try:
        params = {
            "lyric":                  ".",
            "descriptions":           "electronic, calm, ambient",
            "auto_prompt_audio_type": "Electronic",
            "generate_type":          "bgm",
        }
        log.info("Warmup: submitting dummy BGM request")
        rid, cq = await scheduler.submit(params, "warmup")
        t0 = time.time()
        try:
            while True:
                item = await asyncio.wait_for(cq.get(), timeout=900)
                if item is None:
                    break
                if isinstance(item, Exception):
                    log.warning("Warmup error: %s", item)
                    break
        finally:
            scheduler.pending.pop(rid, None)
        log.info("Warmup complete in %.1fs", time.time() - t0)
    except Exception:
        log.exception("Warmup failed (non-fatal)")


@app.on_event("startup")
async def startup():
    scheduler.available_lm   = asyncio.Queue()
    scheduler.request_queue  = asyncio.Queue()
    log.info("Starting pipeline server  LM GPUs=%s  Diff GPUs=%s", LM_GPU_IDS, DIFF_GPU_IDS)
    await scheduler.start_workers_parallel()
    asyncio.create_task(scheduler.scheduler_loop())
    asyncio.create_task(scheduler.result_listener())
    asyncio.create_task(scheduler.watchdog_loop())
    if os.environ.get("SONGGEN_SKIP_WARMUP", "").strip() != "1":
        asyncio.create_task(_warmup())
    log.info("Server ready")


@app.on_event("shutdown")
async def shutdown():
    scheduler.shutdown()


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    lyric: str = Field(..., description=(
        "Song lyrics. Use [verse]/[chorus]/[bridge] etc. "
        "Separate sections with ; and sentences with ."
    ))
    descriptions: Optional[str] = Field(None, description=(
        "Comma-separated style tags, e.g. 'electronic, energetic'."
    ))
    auto_prompt_audio_type: Optional[str] = Field(None, description=(
        "Pop, Latin, Rock, Electronic, Metal, Country, R&B/Soul, Ballad, "
        "Jazz, World, Hip-Hop, Funk, Soundtrack, Auto."
    ))
    generate_type: str = Field("bgm", description=(
        "'bgm' (instrumental), 'mixed' (vocals+bgm), 'vocal'."
    ))


class AudioSetting(BaseModel):
    sample_rate: int = Field(44100, description="Output sample rate in Hz (resampled if different from model native)")
    bitrate: int = Field(128000, description="Target MP3 bitrate in bps (e.g. 128000, 256000). Ignored for WAV.")
    format: str = Field("mp3", description="Output format: 'mp3' or 'wav'")


class MiniMaxRequest(BaseModel):
    model: str = Field("SongGeneration-v2-large", description="Model identifier (informational; server always runs SongGeneration-v2-large)")
    prompt: Optional[str] = Field(None, description="Style prompt, e.g. 'Indie folk, melancholic'. Maps to descriptions.")
    lyrics: Optional[str] = Field(None, description="Song lyrics with [verse], [chorus], etc. Omit for bgm mode.")
    audio_setting: Optional[AudioSetting] = Field(None, description="Output audio format settings")
    auto_prompt_audio_type: Optional[str] = Field(None, description=(
        "Pop, Latin, Rock, Electronic, Metal, Country, R&B/Soul, Ballad, "
        "Jazz, World, Hip-Hop, Funk, Soundtrack, Auto."
    ))
    generate_type: Optional[str] = Field(None, description=(
        "'bgm', 'mixed', or 'vocal'. Default: 'mixed' when lyrics provided, 'bgm' otherwise."
    ))


def _reencode_mp3_chunks(mp3_data: bytes, setting: Optional[AudioSetting]) -> tuple[bytes, str]:
    """Decode concatenated MP3 bytes and re-encode with AudioSetting. Returns (bytes, media_type)."""
    fmt = (setting.format if setting else "mp3").lower()
    target_sr = setting.sample_rate if setting else None
    bitrate = setting.bitrate if setting else 128000

    if fmt == "mp3" and target_sr is None:
        kbps = max(32, min(320, bitrate // 1000))
        if kbps == 128:
            return mp3_data, "audio/mpeg"  # default — no re-encode needed

    wav_cpu, native_sr = torchaudio.load(io.BytesIO(mp3_data), format="mp3")
    if target_sr and target_sr != native_sr:
        wav_cpu = torchaudio.functional.resample(wav_cpu, native_sr, target_sr)
        out_sr = target_sr
    else:
        out_sr = native_sr

    buf = io.BytesIO()
    if fmt == "wav":
        torchaudio.save(buf, wav_cpu, out_sr, format="wav")
        return buf.getvalue(), "audio/wav"

    torchaudio.save(buf, wav_cpu, out_sr, format="mp3")
    return buf.getvalue(), "audio/mpeg"


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else request.client.host


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    lm_alive   = sum(1 for p in scheduler.lm_processes if p.is_alive())
    diff_alive = sum(1 for p in scheduler.diff_processes if p.is_alive())
    return {
        "status":      "ok" if scheduler._started and lm_alive > 0 and diff_alive > 0
                       else "degraded",
        "model":         MODEL_ID,
        "server_type":   "pipeline",
        "lm_gpus":       LM_GPU_IDS,
        "diff_gpus":     DIFF_GPU_IDS,
        "lm_workers_alive":   lm_alive,
        "diff_workers_alive": diff_alive,
        "lm_available":  scheduler.available_lm.qsize() if scheduler.available_lm else 0,
        "queue_size":    scheduler.request_queue.qsize() if scheduler.request_queue else 0,
        "pending":       len(scheduler.pending),
        "diff_queue":    scheduler.diff_queue.qsize(),
        "stats": {
            "total_batches":  scheduler.total_batches,
            "total_requests": scheduler.total_requests,
        },
    }


@app.get("/usage")
def get_usage(n: int = 50):
    log_path = LOG_DIR / "api_usage.log"
    if not log_path.exists():
        return []
    lines = log_path.read_text().splitlines()
    out = []
    for ln in lines[-n:]:
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


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


def _strip_xing_frame(mp3: bytes) -> bytes:
    """Drop the leading Xing/Info VBR-tag MPEG frame if present.
    torchaudio.save(format="mp3") prepends one silent Xing frame per chunk,
    whose frame-count field reflects only that chunk. Concatenating them
    confuses players (e.g. ExoPlayer) that trust Xing and stop early.
    """
    if len(mp3) < 4:
        return mp3
    # Find first MPEG sync (0xFF Ex/Fx)
    for i in range(min(len(mp3) - 4, 32)):
        if mp3[i] == 0xFF and (mp3[i + 1] & 0xE0) == 0xE0:
            h = (mp3[i] << 24) | (mp3[i + 1] << 16) | (mp3[i + 2] << 8) | mp3[i + 3]
            version = (h >> 19) & 0x3   # 11=MPEG1, 10=MPEG2, 00=MPEG2.5
            layer   = (h >> 17) & 0x3   # 01=Layer3
            br_idx  = (h >> 12) & 0xF
            sr_idx  = (h >> 10) & 0x3
            pad     = (h >> 9) & 0x1
            if layer != 0b01 or br_idx in (0, 0xF) or sr_idx == 0x3:
                return mp3
            mpeg1_l3_br = [0,32,40,48,56,64,80,96,112,128,160,192,224,256,320,0]
            mpeg2_l3_br = [0,8,16,24,32,40,48,56,64,80,96,112,128,144,160,0]
            mpeg1_sr    = [44100,48000,32000,0]
            mpeg2_sr    = [22050,24000,16000,0]
            if version == 0b11:
                br = mpeg1_l3_br[br_idx] * 1000; sr = mpeg1_sr[sr_idx]; spf = 1152
            elif version == 0b10:
                br = mpeg2_l3_br[br_idx] * 1000; sr = mpeg2_sr[sr_idx]; spf = 576
            else:
                return mp3
            if br == 0 or sr == 0:
                return mp3
            frame_size = (spf // 8) * br // sr + pad
            frame = mp3[i:i + frame_size]
            if b'Xing' in frame or b'Info' in frame:
                return mp3[:i] + mp3[i + frame_size:]
            return mp3
    return mp3


# Single-byte null heartbeat. MP3 decoders (ExoPlayer, ffmpeg, etc.) scan
# forward for the MPEG sync word (0xFF Ex/Fx) and tolerate stray null bytes
# between frames, so these bytes are safe to inject into the stream. Their
# purpose is to get response headers + an initial HTTP chunk on the wire so
# Cloudflare (and other proxies) stop counting toward the 524 "origin
# timeout" limit while the LM pipeline is still producing chunk 0.
HEARTBEAT_BYTE     = b"\x00"
HEARTBEAT_INTERVAL = 25.0   # seconds between heartbeats while waiting for chunk 0
STREAM_TIMEOUT     = 600.0  # total per-request deadline


async def _stream_request(req: GenerateRequest, ip: str):
    """Core async generator shared by /generate and /generate_stream.

    Heartbeat: yields HEARTBEAT_BYTE every HEARTBEAT_INTERVAL seconds until
    the first real MP3 chunk arrives. Buffered consumers (/generate and
    /v1/music_generation) filter these out by checking `chunk == HEARTBEAT_BYTE`
    before writing to their accumulation buffer.
    """
    params = {
        "lyric":                  req.lyric,
        "descriptions":           req.descriptions,
        "auto_prompt_audio_type": req.auto_prompt_audio_type,
        "generate_type":          req.generate_type,
    }
    rid, cq = await scheduler.submit(params, ip)
    t0 = time.time()
    chunks_out = 0
    bytes_out  = 0
    status     = "ok"
    first_real_received = False
    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= STREAM_TIMEOUT:
                status = "error: timeout"
                raise HTTPException(504, f"Streaming timed out after {STREAM_TIMEOUT:.0f}s")
            # Short wait before chunk 0 so we can emit heartbeats; longer wait
            # afterwards (diff chunks arrive every ~2s once decoding starts).
            if not first_real_received:
                wait = min(HEARTBEAT_INTERVAL, STREAM_TIMEOUT - elapsed)
            else:
                wait = min(60.0, STREAM_TIMEOUT - elapsed)
            try:
                item = await asyncio.wait_for(cq.get(), timeout=wait)
            except asyncio.TimeoutError:
                if not first_real_received:
                    yield HEARTBEAT_BYTE
                # Either way, loop and re-check deadline.
                continue
            if item is None:
                break
            if isinstance(item, Exception):
                status = f"error: {item}"
                raise item
            first_real_received = True
            item = _strip_xing_frame(item)
            chunks_out += 1
            bytes_out  += len(item)
            yield item
    finally:
        # Drop the pending entry so late-arriving diff results (e.g. after a
        # client disconnect) are discarded by result_listener instead of
        # queueing forever against an abandoned asyncio.Queue.
        pr = scheduler.pending.pop(rid, None)
        if pr is not None:
            pr.reorder_buffer.clear()
        log_usage({
            "endpoint":               "/generate_stream",
            "client_ip":              ip,
            "lyric_length":           len(req.lyric),
            "descriptions":           req.descriptions,
            "auto_prompt_audio_type": req.auto_prompt_audio_type,
            "generate_type":          req.generate_type,
            "generation_time_s":      round(time.time() - t0, 2),
            "output_bytes":           bytes_out,
            "chunks":                 chunks_out,
            "status":                 status,
        })


@app.post(
    "/v1/music_generation",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}, "audio/wav": {}}}},
    summary="MiniMax-compatible music generation endpoint",
    dependencies=[Depends(_check_auth)],
)
async def music_generation(req: MiniMaxRequest, request: Request):
    if not scheduler._started:
        raise HTTPException(503, "Workers not ready")

    has_lyrics = bool(req.lyrics and req.lyrics.strip() and req.lyrics.strip() != ".")
    if req.generate_type:
        if req.generate_type not in ("mixed", "vocal", "bgm"):
            raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}")
        generate_type = req.generate_type
    else:
        generate_type = "mixed" if has_lyrics else "bgm"

    lyric = req.lyrics if has_lyrics else "."
    ip = _client_ip(request)
    log.info("[%s] /v1/music_generation lyric_len=%d prompt=%r apt=%r type=%s",
             ip, len(lyric), req.prompt, req.auto_prompt_audio_type, generate_type)

    gen_req = GenerateRequest(
        lyric=lyric,
        descriptions=req.prompt,
        auto_prompt_audio_type=req.auto_prompt_audio_type,
        generate_type=generate_type,
    )

    t0 = time.time()
    buf = io.BytesIO()
    status = "ok"
    chunks_out = 0
    try:
        async for chunk in _stream_request(gen_req, ip):
            if chunk == HEARTBEAT_BYTE:
                continue
            buf.write(chunk)
            chunks_out += 1
    except Exception as exc:
        status = f"error: {exc}"
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
            "chunks": chunks_out,
            "status": status,
        })

    audio_bytes, media_type = _reencode_mp3_chunks(buf.getvalue(), req.audio_setting)
    ext = "wav" if media_type == "audio/wav" else "mp3"
    log.info("[%s] /v1/music_generation done %.1fs %d bytes", ip, elapsed, len(audio_bytes))
    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=songgen_output.{ext}",
            "X-Generation-Time": str(round(elapsed, 2)),
        },
    )


@app.post(
    "/generate_stream",
    summary="Streaming: yields MP3 chunks as audio is decoded (low TTFB)",
    dependencies=[Depends(_check_auth)],
)
async def generate_stream(req: MiniMaxRequest, request: Request):
    if not scheduler._started:
        raise HTTPException(503, "Workers not ready")

    has_lyrics = bool(req.lyrics and req.lyrics.strip() and req.lyrics.strip() != ".")
    if req.generate_type:
        if req.generate_type not in ("mixed", "vocal", "bgm"):
            raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}")
        generate_type = req.generate_type
    else:
        generate_type = "mixed" if has_lyrics else "bgm"

    lyric = req.lyrics if has_lyrics else "."
    ip = _client_ip(request)
    log.info("[%s] /generate_stream lyric_len=%d prompt=%r apt=%r type=%s",
             ip, len(lyric), req.prompt, req.auto_prompt_audio_type, generate_type)

    gen_req = GenerateRequest(
        lyric=lyric,
        descriptions=req.prompt,
        auto_prompt_audio_type=req.auto_prompt_audio_type,
        generate_type=generate_type,
    )
    return StreamingResponse(
        _stream_request(gen_req, ip),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=pipeline_stream.mp3",
            "Cache-Control":       "no-cache, no-store",
            "X-Stream-Mode":       "pipeline",
        },
    )


@app.post(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"audio/mpeg": {}}}},
    summary="Blocking: waits for full song then returns complete MP3",
    dependencies=[Depends(_check_auth)],
)
async def generate(req: GenerateRequest, request: Request):
    if not scheduler._started:
        raise HTTPException(503, "Workers not ready")
    if req.generate_type not in ("mixed", "vocal", "bgm"):
        raise HTTPException(400, f"Invalid generate_type: {req.generate_type!r}")
    ip = _client_ip(request)
    log.info("[%s] /generate lyric_len=%d desc=%r apt=%r type=%s",
             ip, len(req.lyric), req.descriptions,
             req.auto_prompt_audio_type, req.generate_type)
    t0 = time.time()
    buf = io.BytesIO()
    async for chunk in _stream_request(req, ip):
        if chunk == HEARTBEAT_BYTE:
            continue
        buf.write(chunk)
    audio = buf.getvalue()
    elapsed = time.time() - t0
    log.info("[%s] /generate done %.1fs %d bytes", ip, elapsed, len(audio))
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "attachment; filename=songgen_output.mp3",
            "X-Generation-Time":   str(round(elapsed, 2)),
        },
    )
