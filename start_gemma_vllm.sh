#!/bin/bash
CONDA_SH="/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh"
WORKDIR="/cephfs/volumes/hpc_data_usr/k1810895/8a1a0d1a-60bb-4617-8d51-f74c93f2c303/musicgen"

# Always use GPU 0 for Gemma — called only when N_GPUS >= 2 by start_all.sh.
GEMMA_GPU=${GEMMA_GPU:-0}

nohup bash -c "source $CONDA_SH && CUDA_VISIBLE_DEVICES=$GEMMA_GPU conda run -n vllmenv --no-capture-output vllm serve google/gemma-4-E4B-it --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.9 --enable-prefix-caching --trust-remote-code --enforce-eager" >> "$WORKDIR/logs/vllm_gemma_stdout.log" 2>&1 &

echo "Started gemma vLLM server on GPU $GEMMA_GPU (PID $!). Logs: logs/vllm_gemma_stdout.log"
