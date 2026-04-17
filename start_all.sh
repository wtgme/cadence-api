#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$DIR/start_songgen_pipeline.sh"
bash "$DIR/start_gemma_vllm.sh"
bash "$DIR/start_tunnel.sh"
