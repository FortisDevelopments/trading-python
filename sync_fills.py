#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Reuse your timestamp normalizer if available
try:
    from logger_api import utc_mysql_datetime_ms
except Exception:
    def utc_mysql_datetime_ms(dt: datetime) -> str:
        dt = dt.astimezone(timezone.utc)
        ms = int(dt.microsecond / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"


# -----------------------
# Config
# -----------------------
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
BOT_ID = os.getenv("BOT_ID", "btc_4h_live_usdt_1304")
SYMBOL = os.getenv("FILLS_SYMBOL", os.getenv("SYMBOL", "BTCUSDT"))
LOOKBACK_MINUTES = int(os.getenv("FILLS_LOOKBACK_MINUTES", "240"))  # fetch last N minutes
RETRIES = int(os.getenv("API_RETRIES", "3"))
TIMEOUT = float(os.getenv("API_TIMEOUT", "10"))

API_KEY = os.getenv("BINANCE_LIVE_API_KEY")
API_SECRET = os.getenv("BINANCE_LIVE_API_SECRET")


# Local dedupe state file (simple + reliable)
STATE_PATH = os.getenv("FILLS_STATE_PATH", f"fills_state_{SYMBOL}.json")


def headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def post_fill(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")
    url = f"{API_BASE_URL}/api/bot/fills"

    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, json=payload, headers=headers(), timeout=TIMEOUT)
            if 200 <= r.status_code < 300:
                return r.json() if r.content else {"success": True}
            last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        except Exception as e:
            last_err = e
        time.sleep(1.5 * attempt)

    raise last_err


def load_state() -> Dict[str, Any]:
    """
    Keeps a high-water mark so we only send new fills.
    We'll store last_trade_time_ms and (optionally) a set of recent tradeIds.
    """
    if not os.path.exists(STATE_PATH):
        return {"last_trade_time_ms": 0, "recent_trade_ids": []}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_trade_time_ms": 0, "recent_trade_ids": []}


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f)


def trade_to_payload(t: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Binance get_my_trades() item -> your backend fill payload.
    """
    trade_time_ms = int(t["time"])
    trade_dt = datetime.fromtimestamp(trade_time_ms / 1000, tz=timezone.utc)

    is_buyer = bool(t.get("isBuyer", False))
    side = "BUY" if is_buyer else "SELL"

    qty = float(t["qty"])
    price = float(t["price"])
    quote_qty = float(t.get("quoteQty", qty * price))

    payload = {
        "bot_id": BOT_ID,
        "symbol": t.get("symbol", SYMBOL),

        "exchange_trade_id": int(t["id"]),
        "exchange_order_id": int(t["orderId"]) if "orderId" in t else None,

        "side": side,
        "qty": qty,
        "price": price,
        "quote_qty": quote_qty,

        "commission": float(t.get("commission", 0.0)) if t.get("commission") is not None else None,
        "commission_asset": t.get("commissionAsset"),

        "is_maker": int(bool(t.get("isMaker", False))),
        "is_buyer": int(is_buyer),

        "trade_time": utc_mysql_datetime_ms(trade_dt),

        # you said raw_json can be empty for now
        "raw_json": None,
    }
    return payload


def main() -> int:
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing BINANCE_LIVE_API_KEY / BINANCE_LIVE_API_SECRET in .env")

    client = Client(API_KEY, API_SECRET)

    # Determine startTime for Binance API (ms)
    now_ms = int(time.time() * 1000)
    lookback_ms = LOOKBACK_MINUTES * 60 * 1000
    start_time_ms = now_ms - lookback_ms

    state = load_state()
    last_trade_time_ms = int(state.get("last_trade_time_ms", 0))
    # Use max of stored watermark and lookback window
    start_time_ms = max(start_time_ms, last_trade_time_ms)

    # Fetch trades
    try:
        trades: List[Dict[str, Any]] = client.get_my_trades(symbol=SYMBOL, startTime=start_time_ms)
    except BinanceAPIException as e:
        print("Binance API error:", str(e))
        return 1

    if not trades:
        print(f"No trades found for {SYMBOL} since {start_time_ms}.")
        return 0

    # Local dedupe: keep a small list of recent trade IDs
    recent_ids = set(state.get("recent_trade_ids", []))

    # Sort ascending by time so we update watermark safely
    trades_sorted = sorted(trades, key=lambda x: int(x["time"]))

    sent = 0
    max_time_seen = last_trade_time_ms
    new_recent_ids: List[int] = list(recent_ids)

    for t in trades_sorted:
        trade_id = int(t["id"])
        trade_time_ms = int(t["time"])
        max_time_seen = max(max_time_seen, trade_time_ms)

        if trade_id in recent_ids:
            continue

        payload = trade_to_payload(t)

        # POST to backend (backend should also dedupe on unique key)
        resp = post_fill(payload)
        sent += 1
        print(f"Sent fill trade_id={trade_id} side={payload['side']} quote_qty={payload['quote_qty']:.4f} resp={resp.get('success', True)}")

        recent_ids.add(trade_id)
        new_recent_ids.append(trade_id)

    # Keep only last ~500 ids to keep file small
    new_recent_ids = new_recent_ids[-500:]

    # Advance watermark slightly past max time seen (avoid re-fetch loops)
    state["last_trade_time_ms"] = int(max_time_seen) + 1
    state["recent_trade_ids"] = new_recent_ids
    save_state(state)

    print(f"Done. Trades fetched={len(trades_sorted)} sent={sent} watermark={state['last_trade_time_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
