#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from binance.client import Client

# We reuse your logger timestamp helper if available
try:
    from logger_api import utc_mysql_datetime_ms
except Exception:
    def utc_mysql_datetime_ms(dt):
        dt = dt.astimezone(timezone.utc)
        ms = int(dt.microsecond / 1000)
        return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{ms:03d}"


# -----------------------
# Config
# -----------------------
load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
BOT_ID = os.getenv("BOT_ID", "btc_4h_LIVE")

SYMBOL = os.getenv("EQUITY_SYMBOL", "BTCUSDT")  # keep BTCUSDC for now

API_KEY = os.getenv("BINANCE_LIVE_API_KEY")
API_SECRET = os.getenv("BINANCE_LIVE_API_SECRET")

TIMEOUT = float(os.getenv("API_TIMEOUT", "10"))
RETRIES = int(os.getenv("API_RETRIES", "3"))


def headers():
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def post_equity_snapshot(payload: dict) -> dict:
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")

    url = f"{API_BASE_URL}/api/bot/equity-snapshots"
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


def get_totals(client: Client, asset: str) -> tuple[float, float, float]:
    """
    Returns (free, locked, total) for an asset.
    """
    acct = client.get_account()
    for b in acct["balances"]:
        if b["asset"] == asset:
            free = float(b["free"])
            locked = float(b["locked"])
            return free, locked, free + locked
    return 0.0, 0.0, 0.0


def get_price(client: Client, symbol: str) -> float:
    """
    Uses last traded price. Good enough for equity snapshots.
    """
    return float(client.get_symbol_ticker(symbol=symbol)["price"])


def main():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("Missing BINANCE_LIVE_API_KEY / BINANCE_LIVE_API_SECRET in .env")

    client = Client(API_KEY, API_SECRET)

    ts = utc_mysql_datetime_ms(datetime.now(timezone.utc))

    # BTC + USDC totals include locked (e.g., in OCO orders)
    btc_free, btc_locked, btc_total = get_totals(client, "BTC")
    usdt_free, usdt_locked, usdt_total = get_totals(client, "USDT")



    

    price = get_price(client, SYMBOL)
    equity_usdt = usdt_total + btc_total * price

    payload = {
    "bot_id": BOT_ID,
    "ts": ts,
    "symbol": SYMBOL,

    "btc_free": btc_free,
    "btc_locked": btc_locked,
    "btc_total": btc_total,

    "usdt_free": usdt_free,
    "usdt_locked": usdt_locked,
    "usdt_total": usdt_total,

    "mark_price": price,
    "equity_usdt": equity_usdt,
}

    resp = post_equity_snapshot(payload)
    print("Equity snapshot payload:", payload)
    print("API response:", resp)


if __name__ == "__main__":
    main()
