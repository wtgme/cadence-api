#!/bin/bash
CONDA_SH="/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh"
WORKDIR="/users/k1810895/data/musicgen"

# Split LM/Diff across GPUs 0-3 (GPUs 4-5 reserved for vLLM).
# LM=0,1 Diff=2,3 → disjoint sets → token_callback streaming ON → first MP3 bytes ~20-30s.
# Capacity: 2 concurrent LM batches (2 songs each if SONGGEN_BATCH_MAX=2) + 2 diff workers.
nohup bash -c "source $CONDA_SH && cd $WORKDIR && SONGGEN_LM_GPU_IDS=0,1 SONGGEN_DIFF_GPU_IDS=2,3 conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_pipeline_server:app --host 0.0.0.0 --port 8888" >> "$WORKDIR/logs/pipeline_stdout.log" 2>&1 &

echo "Started SongGen pipeline server (PID $!). Logs: logs/pipeline_stdout.log"
