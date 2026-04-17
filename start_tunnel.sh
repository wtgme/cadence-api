#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/.env" ] && set -a && . "$DIR/.env" && set +a

if [ -z "${TUNNEL_TOKEN:-}" ]; then
  echo "TUNNEL_TOKEN not set. Put it in $DIR/.env (TUNNEL_TOKEN=...) or export it." >&2
  exit 1
fi

tmux kill-session -t tunnel 2>/dev/null
tmux new-session -d -s tunnel "cloudflared tunnel run --token $TUNNEL_TOKEN 2>&1 | tee /tmp/tunnel.log"
sleep 5
grep -q "Registered tunnel" /tmp/tunnel.log && echo "Tunnel is UP: https://api.cadencemusics.uk" || echo "Tunnel failed — check /tmp/tunnel.log"
