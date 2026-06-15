#!/bin/bash
CONDA_SH="/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh"
WORKDIR="/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen"

# Always use GPU 0 for Gemma — called only when N_GPUS >= 2 by start_all.sh.
GEMMA_GPU=${GEMMA_GPU:-0}

# Default: legacy google/gemma-4-E4B-it on the `vllmenv` env (vLLM 0.19.0).
# To serve the newer gemma-4-12B-it (a `gemma4_unified` text+vision+audio model that
# only vLLM NIGHTLY supports — released vLLM <=0.22.1 crashes), run with:
#   GEMMA_ENV=gemma4 GEMMA_MODEL=google/gemma-4-12B-it bash start_gemma_vllm.sh
# The `gemma4` env holds vLLM nightly (cu129); the 12B-only workarounds below are applied
# automatically when GEMMA_MODEL is a *12B* model.
# The server still advertises GEMMA_SERVED_NAME (default google/gemma-4-E4B-it) as the API
# model name, so existing clients keep working unchanged even when the 12B weights load.
GEMMA_ENV=${GEMMA_ENV:-vllmenv}
GEMMA_MODEL=${GEMMA_MODEL:-google/gemma-4-E4B-it}

# Expose a STABLE API name regardless of which checkpoint actually loads. vLLM's
# --served-model-name makes the engine advertise this alias in /v1/models and accept
# it as the request "model" field, while loading the real GEMMA_MODEL weights. This lets
# us swap in gemma-4-12B-it without clients having to change the model name they send.
GEMMA_SERVED_NAME=${GEMMA_SERVED_NAME:-google/gemma-4-E4B-it}

COMMON_FLAGS="--port 8000 --served-model-name $GEMMA_SERVED_NAME --max-model-len 16384 --gpu-memory-utilization 0.9 --enable-prefix-caching --trust-remote-code --enforce-eager"

EXTRA_ENV=""
EXTRA_FLAGS=""
if [[ "$GEMMA_MODEL" == *12B* ]]; then
  # CUDA toolkit for FlashInfer JIT: the login env puts CUDA 11.8 on PATH, but FlashInfer's
  # bundled cccl requires CUDA >=12 — point at spack CUDA 12.2 (host gcc 11.4 compatible).
  # VLLM_USE_FLASHINFER_SAMPLER=0 routes top-k/top-p sampling through native PyTorch, avoiding
  # a fragile FlashInfer JIT build of the sampler kernel at engine startup.
  CUDA12_HOME=${CUDA12_HOME:-/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-zen4/gcc-13.2.0/cuda-12.2.1-k46nrhwopvi5zfrnp2ckit4quy6lir53}
  EXTRA_ENV="export CUDA_HOME=$CUDA12_HOME && export PATH=$CUDA12_HOME/bin:\$PATH && export VLLM_USE_FLASHINFER_SAMPLER=0 &&"
  # Text-only ({"image":0,"audio":0}) — skips multimodal overhead for the chat backend.
  EXTRA_FLAGS="--limit-mm-per-prompt '{\"image\": 0, \"audio\": 0}'"
fi

nohup bash -c "source $CONDA_SH && $EXTRA_ENV CUDA_VISIBLE_DEVICES=$GEMMA_GPU conda run -n $GEMMA_ENV --no-capture-output vllm serve $GEMMA_MODEL $COMMON_FLAGS $EXTRA_FLAGS" >> "$WORKDIR/logs/vllm_gemma_stdout.log" 2>&1 &

echo "Started gemma vLLM server (weights=$GEMMA_MODEL, served as '$GEMMA_SERVED_NAME', env=$GEMMA_ENV) on GPU $GEMMA_GPU (PID $!). Logs: logs/vllm_gemma_stdout.log"
