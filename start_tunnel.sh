#!/bin/bash
# Start the Cloudflare tunnel in a detached tmux session.
# Waits until all 4 HA edge connections register so non-LHR PoPs don't 502.

set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$DIR/.env" ] && set -a && . "$DIR/.env" && set +a

if [ -z "${TUNNEL_TOKEN:-}" ]; then
  echo "TUNNEL_TOKEN not set. Put it in $DIR/.env (TUNNEL_TOKEN=...) or export it." >&2
  exit 1
fi

LOG=/tmp/tunnel.log
HOSTS=("https://api.cadencemusics.uk" "https://chat.cadencemusics.uk")
REQUIRED_CONNS=4
WAIT_SECS=60

# Kill previous tmux session AND any stray cloudflared process — orphans
# steal connections from the new run and leave it stuck at 1/4.
tmux kill-session -t tunnel 2>/dev/null
pkill -x cloudflared 2>/dev/null
sleep 1

: > "$LOG"
tmux new-session -d -s tunnel "cloudflared tunnel --loglevel info run --token $TUNNEL_TOKEN 2>&1 | tee $LOG"

count_conns() {
  local n
  n=$(grep -c "Registered tunnel connection" "$LOG" 2>/dev/null)
  echo "${n:-0}"
}

# Poll until we see 4 "Registered tunnel connection" lines, or timeout.
for i in $(seq 1 "$WAIT_SECS"); do
  conns=$(count_conns)
  if [ "$conns" -ge "$REQUIRED_CONNS" ]; then
    echo "Tunnel UP — $conns/$REQUIRED_CONNS edge connections registered"
    for h in "${HOSTS[@]}"; do echo "  $h"; done
    grep -E "Connection .* registered|location=[A-Z]+" "$LOG" | tail -n 8
    exit 0
  fi
  sleep 1
done

conns=$(count_conns)
echo "Tunnel NOT healthy: only $conns/$REQUIRED_CONNS connections after ${WAIT_SECS}s" >&2
echo "Last 20 log lines:" >&2
tail -n 20 "$LOG" >&2
exit 1
