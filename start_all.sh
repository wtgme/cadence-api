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
  GEMMA_ENV=gemma4 GEMMA_MODEL=google/gemma-4-12B-it bash "$DIR/start_gemma_vllm.sh"
  # 12B startup is ~7 min (CephFS weight load + engine init + multimodal warmup), so wait
  # up to 10 min for the socket before starting songgen.
  echo "Waiting for vLLM to bind :8000..."
  for i in $(seq 1 600); do
    if ss -ltn 2>/dev/null | grep -q ':8000 '; then
      echo "vLLM is listening on :8000 (after ${i}s)."
      break
    fi
    sleep 1
  done
  export PIPELINE_FIRST_GPU=1
else
  echo "Only 1 GPU available — skipping Gemma vLLM, pipeline will use GPU 0."
  export PIPELINE_FIRST_GPU=0
fi

bash "$DIR/start_songgen_pipeline.sh"
bash "$DIR/start_tunnel.sh"
