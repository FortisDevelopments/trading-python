#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="/home/trading-python"
LOG_DIR="$REPO_DIR/logs"
VENV_PY="$REPO_DIR/venv/bin/python"
SCRIPT="$REPO_DIR/horizon_exit_manager.py"
LOCAL_TZ="America/Mexico_City"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  source "$REPO_DIR/.env"
  set +a
fi

BOT_ID="${BOT_ID:-btc_4h_LIVE}"

UTC_ISO="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
LOCAL_FMT="$(TZ="$LOCAL_TZ" date +'%Y-%m-%d %H:%M:%S %Z')"

echo "run script=$(basename "$SCRIPT") bot_id=$BOT_ID utc=$UTC_ISO local(${LOCAL_TZ})=$LOCAL_FMT" \
  >> "$LOG_DIR/horizon_exit_runs.log"

set +e
"$VENV_PY" -u "$SCRIPT" >> "$LOG_DIR/horizon_exit.log" 2>> "$LOG_DIR/horizon_exit.err"
RC=$?
set -e

echo "exit_code=$RC script=$(basename "$SCRIPT") bot_id=$BOT_ID utc=$UTC_ISO" \
  >> "$LOG_DIR/horizon_exit_runs.log"

exit $RC
