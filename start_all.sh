#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
echo "Detected $N_GPUS GPU(s)"

if [ "$N_GPUS" -lt 1 ]; then
  echo "No GPUs detected. Aborting." >&2
  exit 1
fi

if [ "$N_GPUS" -ge 2 ]; then
  # Start vLLM first and wait until it's listening on :8000 before launching songgen.
  # SongGen pegs GPUs at 100% utilisation; if it's running during vLLM warm-up,
  # vLLM's CUDA init / graph capture stalls. Eager mode still benefits from the stagger.
  #
  # Serve gemma-4-12B-it on the vLLM-nightly `gemma4` env, but advertise it under the
  # stable API name google/gemma-4-E4B-it (via --served-model-name in start_gemma_vllm.sh)
  # so existing clients need no changes.
  # Readiness is probed with a real HTTP request, not a port-bind check: any process
  # holding :8000 (e.g. a stray songgen server) satisfies `ss | grep :8000` instantly, so a
  # bind check would report success, skip the wait, and leave a stack with no chat backend.
  if curl -sf -m 5 http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "vLLM already serving on :8000 — reusing it."
  elif ss -ltn 2>/dev/null | grep -q ':8000 '; then
    echo "ERROR: port 8000 is held by a process that is not answering as vLLM." >&2
    ss -ltnp 2>/dev/null | grep ':8000 ' >&2
    echo "Free the port (or stop that process) and re-run. Aborting." >&2
    exit 1
  else
    GEMMA_ENV=gemma4 GEMMA_MODEL=google/gemma-4-12B-it bash "$DIR/start_gemma_vllm.sh"
    # 12B startup is ~7 min (CephFS weight load + engine init + warmup), so allow 10 min.
    echo "Waiting for vLLM to serve on :8000..."
    vllm_ready=0
    for i in $(seq 1 600); do
      if curl -sf -m 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
        echo "vLLM is serving on :8000 (after ${i}s)."
        vllm_ready=1
        break
      fi
      sleep 1
    done
    if [ "$vllm_ready" -ne 1 ]; then
      echo "ERROR: vLLM did not answer on :8000 within 600s — see logs/vllm_gemma_stdout.log" >&2
      echo "Aborting rather than starting a stack with no chat backend." >&2
      exit 1
    fi
  fi
  export PIPELINE_FIRST_GPU=1
else
  echo "Only 1 GPU available — skipping Gemma vLLM, pipeline will use GPU 0."
  export PIPELINE_FIRST_GPU=0
fi

bash "$DIR/start_songgen_pipeline.sh"
bash "$DIR/start_tunnel.sh"
