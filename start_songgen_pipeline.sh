#!/bin/bash
CONDA_SH="/software/spackages_v0_21_prod/apps/linux-ubuntu22.04-icelake/gcc-13.2.0/anaconda3-2022.10-tjkkt6f5oslpe3qj7vrpvqrm7vru4k6e/etc/profile.d/conda.sh"
WORKDIR="/users/k1810895/data/musicgen"

# Free port 8888 if something is already bound (legacy songgen server, crashed run, etc.)
PORT_PID=$(ss -ltnp 2>/dev/null | awk '/:8888[[:space:]]/ {print $NF}' | grep -oP 'pid=\K[0-9]+' | head -n1)
if [ -n "$PORT_PID" ]; then
  echo "Port 8888 held by PID $PORT_PID — killing it."
  kill "$PORT_PID" 2>/dev/null
  sleep 2
fi

# PIPELINE_FIRST_GPU: first GPU index available for the pipeline (set by start_all.sh).
# Default is 1 (assumes GPU 0 is taken by Gemma). Set to 0 when running standalone.
FIRST_GPU=${PIPELINE_FIRST_GPU:-1}
N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
N_AVAIL=$(( N_GPUS - FIRST_GPU ))

if [ "$N_AVAIL" -lt 1 ]; then
  echo "No GPUs available for pipeline (total=$N_GPUS, first_available=$FIRST_GPU). Aborting." >&2
  exit 1
fi

# Split: diff = max(1, N_AVAIL // 6), lm = N_AVAIL - diff (mirrors pipeline server auto-split).
N_DIFF=$(( N_AVAIL / 6 )); [ "$N_DIFF" -lt 1 ] && N_DIFF=1
N_LM=$(( N_AVAIL - N_DIFF ))

if [ "$N_LM" -lt 1 ]; then
  # Only 1 GPU available — LM and Diff share it
  LM_IDS="$FIRST_GPU"
  DIFF_IDS="$FIRST_GPU"
else
  LM_IDS=$(seq "$FIRST_GPU" $(( FIRST_GPU + N_LM - 1 )) | paste -sd,)
  DIFF_IDS=$(seq $(( FIRST_GPU + N_LM )) $(( FIRST_GPU + N_LM + N_DIFF - 1 )) | paste -sd,)
fi

echo "GPUs: $N_GPUS total | pipeline GPUs $FIRST_GPU–$(( N_GPUS - 1 )) | LM: $N_LM (GPUs $LM_IDS) | Diff: $N_DIFF (GPU $DIFF_IDS)"

nohup bash -c "
source $CONDA_SH
cd $WORKDIR
while true; do
    echo \"[\$(date)] Starting pipeline server (LM=$LM_IDS Diff=$DIFF_IDS)...\"
    SONGGEN_COMPILE=1 SONGGEN_LM_GPU_IDS=$LM_IDS SONGGEN_DIFF_GPU_IDS=$DIFF_IDS \
        conda run -n musicgen --no-capture-output python -m uvicorn songgeneration_pipeline_server:app --host 0.0.0.0 --port 8888
    echo \"[\$(date)] Server exited (code \$?) — restarting in 15s...\"
    sleep 15
done
" >> "$WORKDIR/logs/pipeline_stdout.log" 2>&1 &

echo "Started SongGen pipeline server (PID $!). Logs: logs/pipeline_stdout.log"
