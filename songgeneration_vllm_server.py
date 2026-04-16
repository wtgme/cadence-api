"""
SongGeneration v2-large — vLLM-style batched inference server.

Architecture
------------
- Main process : FastAPI on port 8000, batch scheduler, result router
- GPU workers  : One per GPU (separate processes), each loads the full model
- Batching     : Incoming requests queue up; scheduler forms batches of up to
                 MAX_BATCH_SIZE and dispatches to the first available GPU worker.
- API          : Identical to songgeneration_server.py (drop-in replacement).

Start
-----
    conda run -n musicgen --no-capture-output python -m uvicorn \
        songgeneration_vllm_server:app --host 0.0.0.0 --port 8000

Environment variables
---------------------
    SONGGEN_GPU_IDS   Comma-separated GPU ids (default: "0,1")
    SONGGEN_BATCH_MAX Size of each batch             (default: 2)
    SONGGEN_BATCH_WAIT_MS  Max ms to wait for batch  (default: 500)
"""

from __future__ import annotations

import asyncio
import gc
import io
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
import torchaudio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GPU_IDS = [int(x) for x in os.environ.get("SONGGEN_GPU_IDS", "0,1").split(",")]
MAX_BATCH_SIZE = int(os.environ.get("SONGGEN_BATCH_MAX", "2"))
BATCH_WAIT_MS = int(os.environ.get("SONGGEN_BATCH_WAIT_MS", "500"))

SONGGEN_ROOT = Path("/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen/songgeneration")
CKPT_PATH = SONGGEN_ROOT / "ckpt" / "songgeneration_v2_large"

MODEL_ID = "SongGeneration-v2-large"

auto_prompt_types = [
    "Pop", "Latin", "Rock", "Electronic", "Metal", "Country",
    "R&B/Soul", "Ballad", "Jazz", "World", "Hip-Hop", "Funk",
    "Soundtrack", "Auto",
]

# ---------------------------------------------------------------------------
# Logging (main process)
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [main] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "vllm_server.log", maxBytes=10 * 1024 * 1024, backupCount=5
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


# ===================================================================
# GPU Worker — runs in a separate process, one per GPU
# ===================================================================

def _check_language(text: str) -> str:
    import re
    chinese_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    if len(text) == 0:
        return "en"
    return "zh" if chinese_count / len(text) >= 0.2 else "en"


def gpu_worker_fn(gpu_id: int, input_queue: mp.Queue, output_queue: mp.Queue):
    """Worker process.  Loads the model on *one* GPU and serves batches."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Add model paths
    _extra_paths = [
        str(SONGGEN_ROOT / "codeclm" / "tokenizer"),
        str(SONGGEN_ROOT),
        str(SONGGEN_ROOT / "codeclm" / "tokenizer" / "Flow1dVAE"),
    ]
    for p in _extra_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(SONGGEN_ROOT / "third_party" / "hub"))
    os.chdir(SONGGEN_ROOT)

    worker_log = logging.getLogger(f"worker-gpu{gpu_id}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"%(asctime)s %(levelname)s [gpu{gpu_id}] %(message)s"))
    worker_log.addHandler(handler)
    worker_log.setLevel(logging.INFO)

    # ---- Load model ----
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(1)
    np.random.seed(int.from_bytes(os.urandom(4), "big"))

    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
    OmegaConf.register_new_resolver("concat", lambda *x: [xxx for xx in x for xxx in xx], replace=True)
    OmegaConf.register_new_resolver("get_fname", lambda: "server", replace=True)
    OmegaConf.register_new_resolver("load_yaml", lambda x: list(OmegaConf.load(x)), replace=True)

    from codeclm.models import builders, CodecLM

    worker_log.info("Loading %s on cuda (physical GPU %d)…", MODEL_ID, gpu_id)
    t0 = time.time()

    cfg = OmegaConf.load(str(CKPT_PATH / "config.yaml"))
    cfg.lm.use_flash_attn_2 = True
    cfg.mode = "inference"
    sample_rate = cfg.sample_rate

    auto_prompt = torch.load(str(SONGGEN_ROOT / "tools" / "new_auto_prompt.pt"), map_location="cpu")

    audiolm = builders.get_lm_model(cfg, version="v2")
    checkpoint = torch.load(str(CKPT_PATH / "model.pt"), map_location="cpu", mmap=True)
    audiolm_state_dict = {k.replace("audiolm.", ""): v for k, v in checkpoint.items() if k.startswith("audiolm")}
    audiolm.load_state_dict(audiolm_state_dict, strict=False)
    del checkpoint, audiolm_state_dict
    gc.collect()
    audiolm = audiolm.eval().cuda().to(torch.float16)

    sep_tok = builders.get_audio_tokenizer_model_cpu(cfg.audio_tokenizer_checkpoint_sep, cfg)
    device = "cuda:0"
    sep_tok.model.device = device
    sep_tok.model.vae = sep_tok.model.vae.to(device)
    sep_tok.model.model.device = torch.device(device)
    sep_tok.model.model = sep_tok.model.model.to(device)
    sep_tok = sep_tok.eval()
    gc.collect()
    torch.cuda.empty_cache()

    model = CodecLM(
        name=f"songgen_gpu{gpu_id}",
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
    gc.collect()
    torch.cuda.empty_cache()
    worker_log.info("Model ready in %.1fs", time.time() - t0)

    # Signal readiness
    output_queue.put({"type": "ready", "gpu_id": gpu_id})

    # ---- Serve loop ----
    while True:
        batch_msg = input_queue.get()
        if batch_msg is None:
            worker_log.info("Shutdown signal received")
            break

        batch_id = batch_msg["batch_id"]
        requests = batch_msg["requests"]
        B = len(requests)
        worker_log.info("Batch %s: processing %d request(s)", batch_id, B)
        t_batch = time.time()

        try:
            # ---- Prepare inputs for all requests ----
            all_lyrics = []
            all_descriptions = []
            all_pmt_wavs = []
            all_vocal_wavs = []
            all_bgm_wavs = []
            any_has_auto_prompt = False

            for req in requests:
                lyric = req["lyric"]
                descriptions = req.get("descriptions")
                auto_prompt_audio_type = req.get("auto_prompt_audio_type")
                generate_type = req.get("generate_type", "bgm")

                # Build description string (same logic as original server)
                if generate_type == "bgm":
                    desc = "[Musicality-very-high], [Pure-Music], " + (descriptions.lower() if descriptions else ".")
                elif descriptions:
                    desc = "[Musicality-very-high], " + descriptions.lower()
                else:
                    desc = "[Musicality-very-high], ."

                lyric_text = lyric.replace("  ", " ") if generate_type != "bgm" else "."

                # Auto-prompt selection
                if auto_prompt_audio_type and auto_prompt_audio_type in auto_prompt_types:
                    lang = _check_language(lyric)
                    prompts = auto_prompt[auto_prompt_audio_type][lang]
                    prompt_token = prompts[np.random.randint(0, len(prompts))]
                    pmt_wav = prompt_token[:, [0], :]
                    vocal_wav = prompt_token[:, [1], :]
                    bgm_wav = prompt_token[:, [2], :]
                    any_has_auto_prompt = True
                else:
                    pmt_wav = None
                    vocal_wav = None
                    bgm_wav = None

                all_lyrics.append(lyric_text)
                all_descriptions.append(desc)
                all_pmt_wavs.append(pmt_wav)
                all_vocal_wavs.append(vocal_wav)
                all_bgm_wavs.append(bgm_wav)

            # Stack prompt tensors for batched generation.
            # When auto_prompt is used, the prompt tokens are already discrete
            # token indices (not raw waveforms), so melody_is_wav=False.
            # When no auto_prompt, we pass None and _prepare_tokens_and_attributes
            # creates the zero-fill padding internally — melody_is_wav is irrelevant
            # because the None-melody path never calls encode().
            if any_has_auto_prompt:
                # All requests need compatible prompt tensors.
                # Fill missing ones with padding token (16385).
                ref = next(p for p in all_pmt_wavs if p is not None)
                ref_shape = ref.shape  # [1, 1, T]
                for i in range(B):
                    if all_pmt_wavs[i] is None:
                        all_pmt_wavs[i] = torch.full(ref_shape, 16385, dtype=ref.dtype)
                        all_vocal_wavs[i] = torch.full(ref_shape, 16385, dtype=ref.dtype)
                        all_bgm_wavs[i] = torch.full(ref_shape, 16385, dtype=ref.dtype)
                stacked_pmt = torch.cat(all_pmt_wavs, dim=0)      # [B, 1, T]
                stacked_vocal = torch.cat(all_vocal_wavs, dim=0)   # [B, 1, T]
                stacked_bgm = torch.cat(all_bgm_wavs, dim=0)      # [B, 1, T]
                melody_is_wav = False  # prompt tokens are discrete indices
            else:
                stacked_pmt = None
                stacked_vocal = None
                stacked_bgm = None
                melody_is_wav = True  # irrelevant — None melody skips encode

            # ---- Batched token generation ----
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    tokens = model.generate(
                        lyrics=all_lyrics,
                        descriptions=all_descriptions,
                        melody_wavs=stacked_pmt,
                        vocal_wavs=stacked_vocal,
                        bgm_wavs=stacked_bgm,
                        melody_is_wav=melody_is_wav,
                        return_tokens=True,
                    )

            # ---- Decode audio per-sequence and stream results back ----
            # Each decoded sequence is sent immediately so the main process
            # can resolve that client's future without waiting for the rest.
            for i in range(B):
                seq_tokens = tokens[i:i+1]  # [1, K, T]
                gen_type = requests[i].get("generate_type", "bgm")

                # Per-sequence EOS trimming
                eos_mask = (seq_tokens == model.lm.eos_token_id)
                if eos_mask.any():
                    eos_pos = torch.nonzero(eos_mask)[:, -1].min()
                    seq_tokens = seq_tokens[..., :eos_pos]

                with torch.no_grad():
                    wav_mixed = model.generate_audio(
                        seq_tokens, chunked=True, gen_type=gen_type
                    )

                mixed_cpu = wav_mixed[0].cpu().float()
                duration_s = mixed_cpu.shape[-1] / sample_rate

                buf = io.BytesIO()
                torchaudio.save(buf, mixed_cpu, sample_rate, format="mp3")

                # Send this sequence's result immediately
                output_queue.put({
                    "type": "partial_result",
                    "batch_id": batch_id,
                    "result": {
                        "request_id": requests[i]["request_id"],
                        "audio_bytes": buf.getvalue(),
                        "duration_s": duration_s,
                        "status": "ok",
                    },
                })
                worker_log.info(
                    "Batch %s: seq %d/%d decoded (%.1fs audio)",
                    batch_id, i + 1, B, duration_s,
                )
                del wav_mixed, mixed_cpu

            torch.cuda.empty_cache()
            elapsed = time.time() - t_batch
            worker_log.info(
                "Batch %s: done in %.1fs (%d sequences, %.1fs/seq)",
                batch_id, elapsed, B, elapsed / B,
            )
            # Signal batch completion (GPU can be released)
            output_queue.put({
                "type": "batch_done",
                "batch_id": batch_id,
                "elapsed": elapsed,
            })

        except Exception as exc:
            worker_log.exception("Batch %s failed: %s", batch_id, exc)
            gc.collect()
            torch.cuda.empty_cache()
            # Return error for all requests in the batch
            for req in requests:
                output_queue.put({
                    "type": "partial_result",
                    "batch_id": batch_id,
                    "result": {
                        "request_id": req["request_id"],
                        "status": f"error: {exc}",
                        "audio_bytes": None,
                        "duration_s": 0,
                    },
                })
            output_queue.put({
                "type": "batch_done",
                "batch_id": batch_id,
                "elapsed": time.time() - t_batch,
            })


# ===================================================================
# Main process — FastAPI + Batch Scheduler
# ===================================================================

app = FastAPI(title="SongGeneration v2-large vLLM API", version="3.0.0")


@dataclass
class PendingRequest:
    request_id: str
    params: dict
    future: asyncio.Future
    submit_time: float
    client_ip: str


class BatchScheduler:
    """Collects requests and dispatches batches to GPU workers.

    Workers stream results back incrementally (one message per decoded sequence),
    so clients whose sequences finish first get their response immediately.
    """

    def __init__(self, gpu_ids: list[int], max_batch_size: int, max_wait_ms: int):
        self.gpu_ids = gpu_ids
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Per-worker multiprocessing queues
        self.worker_input_queues: list[mp.Queue] = []
        self.worker_output_queues: list[mp.Queue] = []
        self.worker_processes: list[mp.Process] = []

        # Async request queue
        self.request_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()

        # Available GPU tracker (asyncio queue of gpu indices)
        self.available_gpus: asyncio.Queue[int] = asyncio.Queue()

        # Pending futures keyed by request_id
        self.pending: dict[str, asyncio.Future] = {}

        # Active batches: batch_id → {gpu_idx, pending_requests}
        self._active_batches: dict[str, dict] = {}

        # Stats
        self.total_batches = 0
        self.total_requests = 0

        self._started = False

    def start_workers(self):
        """Spawn one process per GPU. Called from FastAPI startup event."""
        mp.set_start_method("spawn", force=True)
        for idx, gpu_id in enumerate(self.gpu_ids):
            in_q = mp.Queue()
            out_q = mp.Queue()
            p = mp.Process(
                target=gpu_worker_fn,
                args=(gpu_id, in_q, out_q),
                daemon=True,
                name=f"gpu-worker-{gpu_id}",
            )
            p.start()
            self.worker_input_queues.append(in_q)
            self.worker_output_queues.append(out_q)
            self.worker_processes.append(p)
            log.info("Started worker process for GPU %d (pid %d)", gpu_id, p.pid)

    async def wait_for_workers_ready(self):
        """Block until all workers signal readiness."""
        loop = asyncio.get_running_loop()
        for idx, out_q in enumerate(self.worker_output_queues):
            msg = await loop.run_in_executor(None, out_q.get)
            assert msg["type"] == "ready", f"Unexpected message from worker: {msg}"
            log.info("Worker GPU %d ready", msg["gpu_id"])
            self.available_gpus.put_nowait(idx)
        self._started = True
        log.info("All %d workers ready — scheduler active", len(self.gpu_ids))

    async def submit_request(self, params: dict, client_ip: str) -> dict:
        """Submit a generation request. Returns result dict when done."""
        request_id = uuid.uuid4().hex[:12]
        params["request_id"] = request_id
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        pending = PendingRequest(
            request_id=request_id,
            params=params,
            future=future,
            submit_time=time.time(),
            client_ip=client_ip,
        )
        self.pending[request_id] = future
        await self.request_queue.put(pending)
        log.info("[%s] Request %s queued (queue size: %d)",
                 client_ip, request_id, self.request_queue.qsize())
        try:
            result = await future
            return result
        finally:
            self.pending.pop(request_id, None)

    async def scheduler_loop(self):
        """Main scheduler loop — collect requests into batches and dispatch."""
        log.info("Scheduler loop started (batch_max=%d, wait_ms=%d)",
                 self.max_batch_size, self.max_wait_ms)
        while True:
            # Wait for the first request (no timeout — idle when no requests)
            first = await self.request_queue.get()
            batch: list[PendingRequest] = [first]

            # Try to collect more requests up to batch size or wait deadline
            deadline = asyncio.get_event_loop().time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    req = await asyncio.wait_for(
                        self.request_queue.get(), timeout=remaining
                    )
                    batch.append(req)
                except asyncio.TimeoutError:
                    break

            self.total_batches += 1
            self.total_requests += len(batch)
            log.info("Formed batch of %d request(s), waiting for GPU…", len(batch))

            # Wait for any GPU to become available
            gpu_idx = await self.available_gpus.get()
            log.info("Dispatching batch to GPU worker %d", gpu_idx)

            # Dispatch asynchronously (don't block scheduler for next batch)
            asyncio.create_task(self._dispatch_batch(batch, gpu_idx))

    async def _dispatch_batch(self, batch: list[PendingRequest], gpu_idx: int):
        """Send batch to worker. Results are handled by the per-worker listener."""
        batch_id = uuid.uuid4().hex[:8]

        # Register active batch so the listener can route results
        pending_by_id = {p.request_id: p for p in batch}
        self._active_batches[batch_id] = {
            "gpu_idx": gpu_idx,
            "pending": pending_by_id,
        }

        batch_msg = {
            "batch_id": batch_id,
            "requests": [p.params for p in batch],
        }

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, self.worker_input_queues[gpu_idx].put, batch_msg
            )
        except Exception as exc:
            log.exception("Failed to send batch to worker %d: %s", gpu_idx, exc)
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(exc)
            self._active_batches.pop(batch_id, None)
            self.available_gpus.put_nowait(gpu_idx)

    async def _start_result_listeners(self):
        """Start one listener task per worker output queue."""
        for idx in range(len(self.worker_output_queues)):
            asyncio.create_task(self._result_listener(idx))

    async def _result_listener(self, worker_idx: int):
        """Listen on a worker's output queue and route results to futures.

        Workers send incremental ``partial_result`` messages (one per decoded
        sequence) followed by a ``batch_done`` message.  Each partial result
        resolves its client's future immediately so the HTTP response goes out
        without waiting for the rest of the batch.

        If the worker process dies, all active batches on that worker are
        failed and the GPU slot is released.
        """
        out_q = self.worker_output_queues[worker_idx]
        proc = self.worker_processes[worker_idx]
        loop = asyncio.get_running_loop()

        def _get_with_timeout():
            """Blocking get with 5s timeout so we can check proc health."""
            import queue
            try:
                return out_q.get(timeout=5)
            except queue.Empty:
                return None

        while True:
            msg = await loop.run_in_executor(None, _get_with_timeout)

            # Check if worker is still alive
            if msg is None:
                if not proc.is_alive():
                    log.error("Worker %d (pid %d) died! Failing active batches.",
                              worker_idx, proc.pid)
                    self._fail_worker_batches(worker_idx,
                                              "GPU worker process crashed")
                    break
                continue  # timeout but worker alive — poll again

            if msg["type"] == "partial_result":
                batch_id = msg["batch_id"]
                result = msg["result"]
                request_id = result["request_id"]
                batch_info = self._active_batches.get(batch_id)
                if batch_info:
                    pending = batch_info["pending"].get(request_id)
                    if pending and not pending.future.done():
                        pending.future.set_result(result)
                        log.info("Request %s resolved (batch %s)", request_id, batch_id)

            elif msg["type"] == "batch_done":
                batch_id = msg["batch_id"]
                batch_info = self._active_batches.pop(batch_id, None)
                if batch_info:
                    gpu_idx = batch_info["gpu_idx"]
                    # Resolve any remaining futures that weren't resolved by partial results
                    for rid, pending in batch_info["pending"].items():
                        if not pending.future.done():
                            pending.future.set_exception(
                                RuntimeError("No result received for request")
                            )
                    # Release GPU for next batch
                    self.available_gpus.put_nowait(gpu_idx)
                    log.info("Batch %s complete, GPU %d released (%.1fs)",
                             batch_id, gpu_idx, msg.get("elapsed", 0))

            elif msg["type"] == "ready":
                pass  # already handled during startup

    def _fail_worker_batches(self, worker_idx: int, error_msg: str):
        """Fail all active batches assigned to a dead worker and release its GPU slot."""
        failed_batch_ids = []
        for batch_id, info in list(self._active_batches.items()):
            if info["gpu_idx"] == worker_idx:
                for rid, pending in info["pending"].items():
                    if not pending.future.done():
                        pending.future.set_exception(RuntimeError(error_msg))
                failed_batch_ids.append(batch_id)
        for bid in failed_batch_ids:
            self._active_batches.pop(bid, None)
        # Do NOT release the GPU slot — the worker is dead and cannot serve.
        # The GPU will be unavailable until the server is restarted.
        log.error("Worker %d is dead. %d batch(es) failed. "
                  "GPU slot removed from pool — restart server to recover.",
                  worker_idx, len(failed_batch_ids))

    def shutdown(self):
        """Gracefully stop all workers."""
        for in_q in self.worker_input_queues:
            in_q.put(None)
        for p in self.worker_processes:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
scheduler = BatchScheduler(
    gpu_ids=GPU_IDS,
    max_batch_size=MAX_BATCH_SIZE,
    max_wait_ms=BATCH_WAIT_MS,
)

# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    log.info("Starting vLLM-style server with GPUs: %s", GPU_IDS)
    scheduler.start_workers()
    await scheduler.wait_for_workers_ready()
    await scheduler._start_result_listeners()
    asyncio.create_task(scheduler.scheduler_loop())
    log.info("Server ready — accepting requests")


@app.on_event("shutdown")
async def shutdown():
    log.info("Shutting down workers…")
    scheduler.shutdown()


# ---------------------------------------------------------------------------
# Request model (identical to original server)
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

def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


@app.get("/health")
def health():
    workers_alive = sum(1 for p in scheduler.worker_processes if p.is_alive())
    gpus_available = scheduler.available_gpus.qsize()
    active_batches = len(scheduler._active_batches)
    return {
        "status": "ok" if scheduler._started and workers_alive > 0 else "starting",
        "model": MODEL_ID,
        "server_type": "vllm-batch",
        "gpus": GPU_IDS,
        "workers_alive": workers_alive,
        "workers_total": len(GPU_IDS),
        "gpus_available": gpus_available,
        "active_batches": active_batches,
        "max_batch_size": MAX_BATCH_SIZE,
        "queue_size": scheduler.request_queue.qsize(),
        "pending_requests": len(scheduler.pending),
        "stats": {
            "total_batches": scheduler.total_batches,
            "total_requests": scheduler.total_requests,
        },
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
    summary="Lyrics + descriptions to song (batched multi-GPU inference)",
)
async def generate(req: GenerateRequest, request: Request):
    if not scheduler._started:
        raise HTTPException(503, "Workers not ready yet.")
    if req.generate_type not in ("mixed", "vocal", "bgm"):
        raise HTTPException(
            400,
            f"Invalid generate_type: {req.generate_type!r}. Use 'bgm', 'mixed', or 'vocal'.",
        )

    ip = client_ip(request)
    log.info(
        "[%s] /generate lyric_len=%d descriptions=%r auto_prompt=%r type=%s",
        ip, len(req.lyric), req.descriptions, req.auto_prompt_audio_type,
        req.generate_type,
    )

    t0 = time.time()
    status = "ok"
    output_bytes = 0

    try:
        result = await asyncio.wait_for(
            scheduler.submit_request(
                params={
                    "lyric": req.lyric,
                    "descriptions": req.descriptions,
                    "auto_prompt_audio_type": req.auto_prompt_audio_type,
                    "generate_type": req.generate_type,
                },
                client_ip=ip,
            ),
            timeout=600,  # 10-minute total timeout
        )

        if result["status"] != "ok":
            status = result["status"]
            raise HTTPException(500, result["status"])

        audio_bytes = result["audio_bytes"]
        output_bytes = len(audio_bytes)

    except asyncio.TimeoutError:
        status = "error: total request timed out (600s)"
        log.error("[%s] /generate timed out after 600s", ip)
        raise HTTPException(504, "Request timed out after 600s")
    except HTTPException:
        raise
    except Exception as exc:
        status = f"error: {exc}"
        log.exception("Generation failed")
        raise HTTPException(500, str(exc))
    finally:
        elapsed = time.time() - t0
        log_usage({
            "endpoint": "/generate",
            "server_type": "vllm-batch",
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
