#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="/home/trading-python"
LOG_DIR="$REPO_DIR/logs"

VENV_PY="$REPO_DIR/venv/bin/python"

# LIVE script (new)
SCRIPT="$REPO_DIR/api_liveScript.py"

# timezone for the "local" timestamp log
LOCAL_TZ="America/Mexico_City"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

# Load env vars from .env (if you use it)
if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

# Optional: show which bot/script ran in the heartbeat
BOT_ID="${BOT_ID:-btc_4h_live}"

# --- heartbeat log: server UTC + Mexico City local time ---
UTC_ISO="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
LOCAL_FMT="$(TZ="$LOCAL_TZ" date +'%Y-%m-%d %H:%M:%S %Z')"

echo "run script=$(basename "$SCRIPT") bot_id=$BOT_ID utc=$UTC_ISO local(${LOCAL_TZ})=$LOCAL_FMT" \
  >> "$LOG_DIR/runs_live.log"

# --- run script with production logging ---
# capture exit code without breaking set -e
set +e
"$VENV_PY" -u "$SCRIPT" >> "$LOG_DIR/api_live.log" 2>> "$LOG_DIR/api_live.err"
RC=$?
set -e

echo "exit_code=$RC script=$(basename "$SCRIPT") bot_id=$BOT_ID utc=$UTC_ISO" >> "$LOG_DIR/runs_live.log"
exit $RC
