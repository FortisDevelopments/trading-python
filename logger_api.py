# logger_api.py
from __future__ import annotations
import os
import time
import requests
from typing import Optional, Dict, Any

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")   # e.g. http://1.2.3.4:3000
API_TOKEN = os.getenv("API_TOKEN", "")                     # optional bearer token
TIMEOUT = float(os.getenv("API_TIMEOUT", "10"))

from datetime import datetime, timezone
from typing import Union

def utc_mysql_datetime_ms(dt: Union[str, datetime, None]) -> str | None:

    if dt is None:
        return None

    # If already MySQL-ish: "YYYY-MM-DD HH:MM:SS" optionally with ".mmm"
    if isinstance(dt, str):
        s = dt.strip()
        if "T" not in s and len(s) >= 19 and s[4] == "-" and s[7] == "-" and s[10] == " ":
            # ensure .mmm exists if not provided
            if len(s) == 19:
                return s + ".000"
            return s

        # Parse ISO-like strings
        # Handle trailing Z
        s2 = s.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(s2)
        except ValueError:
            return s  # leave as-is if unknown format

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        parsed = parsed.astimezone(timezone.utc)

        ms = int(parsed.microsecond / 1000)
        return parsed.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"

    # datetime input
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
        ms = int(dt.microsecond / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"

    return None


def normalize_run_payload(payload: dict) -> dict:
    """
    Normalizes timestamp fields to MySQL DATETIME(3) if present.
    """
    p = dict(payload)
    if "run_ts" in p:
        p["run_ts"] = utc_mysql_datetime_ms(p["run_ts"])
    if "candle_ts" in p:
        p["candle_ts"] = utc_mysql_datetime_ms(p["candle_ts"])
    return p


def normalize_order_payload(payload: dict) -> dict:
    """
    Normalizes timestamp fields to MySQL DATETIME(3) if present.
    """
    p = dict(payload)
    if "executed_at" in p:
        p["executed_at"] = utc_mysql_datetime_ms(p["executed_at"])
    return p

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h

def _post(path: str, payload: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
    """
    POST JSON with small retry/backoff. Raises on final failure.
    """
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")

    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            r = requests.post(url, json=payload, headers=_headers(), timeout=TIMEOUT)
            if 200 <= r.status_code < 300:
                return r.json() if r.content else {"ok": True}
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            last_err = e

        time.sleep(1.5 * attempt)

    raise last_err

def post_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends bot run record.
    Expected endpoint: POST /api/bot/runs
    """
    return _post("/api/bot/runs", normalize_run_payload(payload))

def post_order(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sends bot order record.
    Expected endpoint: POST /api/bot/orders
    """
    return _post("/api/bot/orders", normalize_order_payload(payload))

def post_run_id(payload: Dict[str, Any]) -> int:
    """
    Calls post_run() and returns the created run id (data.id).
    Raises if the id can't be parsed.
    """
    resp = post_run(payload)
    run_id = (resp or {}).get("data", {}).get("id")
    if not run_id:
        raise RuntimeError(f"Could not parse run_id from response: {resp}")
    return int(run_id)


def log_run_and_optional_order(run_payload: Dict[str, Any], order_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Convenience helper:
      1) POST /api/bot/runs
      2) If order_payload is provided, attach run_id then POST /api/bot/orders

    Returns {run_id, run, order}
    """
    run_resp = post_run(run_payload)
    run_id = (run_resp or {}).get("data", {}).get("id")
    if not run_id:
        raise RuntimeError(f"Could not parse run_id from response: {run_resp}")

    order_resp = None
    if order_payload is not None:
        payload = dict(order_payload)  # don't mutate caller dict
        payload["run_id"] = int(run_id)
        order_resp = post_order(payload)

    return {"run_id": int(run_id), "run": run_resp, "order": order_resp}

