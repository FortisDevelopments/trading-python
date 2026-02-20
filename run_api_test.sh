#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="/home/trading-python"
LOG_DIR="$REPO_DIR/logs"

VENV_PY="$REPO_DIR/venv/bin/python"
SCRIPT="$REPO_DIR/api_testScript.py"

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

# --- heartbeat log: server UTC + Mexico City local time ---
# ISO UTC (server)
UTC_ISO="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
# Local time in Mexico City
LOCAL_FMT="$(TZ="$LOCAL_TZ" date +'%Y-%m-%d %H:%M:%S %Z')"

echo "run utc=$UTC_ISO local(${LOCAL_TZ})=$LOCAL_FMT" >> "$LOG_DIR/runs.log"

# --- run script with production logging ---
"$VENV_PY" -u "$SCRIPT" >> "$LOG_DIR/api_test.log" 2>> "$LOG_DIR/api_test.err"
