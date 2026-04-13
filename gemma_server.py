"""
Gemma 4 — FastAPI inference server.

Endpoints
---------
POST /generate        — text generation (JSON body)
POST /generate-vision — image-conditioned generation (multipart form)
GET  /health          — liveness check
GET  /usage           — tail recent API usage log entries
"""

import io
import json
import logging
import logging.handlers
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_DIR / "gemma_server.log", maxBytes=10 * 1024 * 1024, backupCount=5
        ),
    ],
)
log = logging.getLogger(__name__)

usage_logger = logging.getLogger("gemma_usage")
usage_logger.setLevel(logging.INFO)
usage_logger.propagate = False
_usage_handler = logging.handlers.RotatingFileHandler(
    LOG_DIR / "gemma_api_usage.log", maxBytes=10 * 1024 * 1024, backupCount=10
)
_usage_handler.setFormatter(logging.Formatter("%(message)s"))
usage_logger.addHandler(_usage_handler)


def log_usage(record: dict):
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    usage_logger.info(json.dumps(record))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_ID = "google/gemma-4-26B-A4B-it"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Gemma 4 API", version="1.0.0")

processor: AutoProcessor = None
model: AutoModelForCausalLM = None


@app.on_event("startup")
async def load_model():
    global processor, model
    log.info("Loading %s on %s…", MODEL_ID, DEVICE)
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": "cuda:0"},
    )
    model.eval()
    log.info("Model ready in %.1fs", time.time() - t0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    return forwarded.split(",")[0].strip() if forwarded else request.client.host


def load_image_upload(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def build_chat_inputs(messages: list[dict], images: list[Image.Image] | None = None):
    """Apply chat template and tokenise."""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if images:
        inputs = processor(text=text, images=images, return_tensors="pt")
    else:
        inputs = processor(text=text, return_tensors="pt")
    return inputs.to(DEVICE)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text.")


class GenerateRequest(BaseModel):
    messages: list[Message] = Field(..., description="Conversation history.")
    max_new_tokens: int = Field(512, ge=1, le=8192)
    temperature: float = Field(1.0, ge=0.0, le=5.0)
    top_k: int = Field(50, ge=0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    do_sample: bool = Field(True)


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
    }


@app.get("/usage", summary="Recent API usage log entries")
def get_usage(n: int = 50):
    """Return the last *n* usage log entries as a JSON array."""
    usage_log = LOG_DIR / "gemma_api_usage.log"
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


@app.post("/generate", summary="Text generation")
async def generate(req: GenerateRequest, request: Request):
    if model is None:
        raise HTTPException(503, "Model not loaded yet.")

    ip = client_ip(request)
    log.info("[%s] /generate messages=%d max_new_tokens=%d", ip, len(req.messages), req.max_new_tokens)

    messages = [m.model_dump() for m in req.messages]
    inputs = build_chat_inputs(messages)

    t0 = time.time()
    status = "ok"
    generated_text = ""
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature if req.do_sample else None,
                top_k=req.top_k if req.do_sample else None,
                top_p=req.top_p if req.do_sample else None,
                do_sample=req.do_sample,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        generated_text = processor.decode(new_tokens, skip_special_tokens=True)
    except Exception as exc:
        status = f"error: {exc}"
        log.exception("Generation failed")
        raise HTTPException(500, str(exc))
    finally:
        elapsed = time.time() - t0
        log_usage({
            "endpoint": "/generate",
            "client_ip": ip,
            "num_messages": len(req.messages),
            "max_new_tokens": req.max_new_tokens,
            "temperature": req.temperature,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "do_sample": req.do_sample,
            "generation_time_s": round(elapsed, 2),
            "output_tokens": len(generated_text.split()),
            "status": status,
        })

    log.info("[%s] /generate done in %.1fs", ip, elapsed)
    return {"text": generated_text, "generation_time_s": round(elapsed, 2)}


@app.post("/generate-vision", summary="Image-conditioned text generation")
async def generate_vision(
    request: Request,
    prompt: str = Form(..., description="User text prompt."),
    image: UploadFile = File(..., description="Input image (jpg/png/webp)."),
    max_new_tokens: int = Form(512, ge=1, le=8192),
    temperature: float = Form(1.0, ge=0.0, le=5.0),
    top_k: int = Form(50, ge=0),
    top_p: float = Form(0.95, ge=0.0, le=1.0),
    do_sample: bool = Form(True),
):
    if model is None:
        raise HTTPException(503, "Model not loaded yet.")

    ip = client_ip(request)
    log.info("[%s] /generate-vision prompt=%r max_new_tokens=%d", ip, prompt, max_new_tokens)

    image_data = await image.read()
    pil_image = load_image_upload(image_data)

    messages = [{"role": "user", "content": prompt}]
    inputs = build_chat_inputs(messages, images=[pil_image])

    t0 = time.time()
    status = "ok"
    generated_text = ""
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_k=top_k if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        generated_text = processor.decode(new_tokens, skip_special_tokens=True)
    except Exception as exc:
        status = f"error: {exc}"
        log.exception("Generation failed")
        raise HTTPException(500, str(exc))
    finally:
        elapsed = time.time() - t0
        log_usage({
            "endpoint": "/generate-vision",
            "client_ip": ip,
            "prompt": prompt,
            "image_filename": image.filename,
            "image_size_bytes": len(image_data),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "do_sample": do_sample,
            "generation_time_s": round(elapsed, 2),
            "output_tokens": len(generated_text.split()),
            "status": status,
        })

    log.info("[%s] /generate-vision done in %.1fs", ip, elapsed)
    return {"text": generated_text, "generation_time_s": round(elapsed, 2)}
