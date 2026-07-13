#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List

from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))


# -----------------------
# Config
# -----------------------
BOT_ID = os.getenv("BOT_ID", "btc_4h_LIVE")
SYMBOL = os.getenv("SYMBOL", "BTCUSDC").upper()

RESAMPLE_RULE = os.getenv("RESAMPLE_RULE", "4h").lower()
HORIZON_STEPS = int(os.getenv("HORIZON_STEPS", "6"))

# We want to exit 10 minutes before the next 4H buy cycle.
HORIZON_EXIT_BUFFER_MINUTES = int(os.getenv("HORIZON_EXIT_BUFFER_MINUTES", "10"))

# Tiny tolerance so a buy at 00:00:05 does not miss the 23:50:00 check.
HORIZON_EXIT_GRACE_SECONDS = int(os.getenv("HORIZON_EXIT_GRACE_SECONDS", "90"))

# Safety switch. Uses same live-trading flag as api_liveScript.py.
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "0") == "1"

# Optional cleanup step: sell any free leftover base asset back into quote.
# This never touches locked BTC from active OCO orders.
SWEEP_FREE_BASE_DUST = os.getenv("SWEEP_FREE_BASE_DUST", "1") == "1"
DUST_SWEEP_RESERVED_QTY = float(os.getenv("DUST_SWEEP_RESERVED_QTY", "0"))

API_KEY = os.getenv("BINANCE_LIVE_API_KEY")
API_SECRET = os.getenv("BINANCE_LIVE_API_SECRET")


QUOTE_ASSETS = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "DAI", "USD")


def parse_interval_minutes(rule: str) -> int:
    rule = rule.strip().lower()

    if rule.endswith("m"):
        return int(rule[:-1])

    if rule.endswith("h"):
        return int(rule[:-1]) * 60

    if rule.endswith("d"):
        return int(rule[:-1]) * 24 * 60

    raise ValueError(f"Unsupported RESAMPLE_RULE={rule!r}")


INTERVAL_MINUTES = parse_interval_minutes(RESAMPLE_RULE)

# For 4h × 6 with 10-minute buffer: 1430 minutes = 23h50m.
HORIZON_EXIT_MINUTES = int(
    os.getenv(
        "HORIZON_EXIT_MINUTES",
        str((HORIZON_STEPS * INTERVAL_MINUTES) - HORIZON_EXIT_BUFFER_MINUTES),
    )
)


def infer_base_asset(symbol: str) -> str:
    for quote in QUOTE_ASSETS:
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    raise ValueError(f"Could not infer base asset from symbol={symbol}")


BASE_ASSET = infer_base_asset(SYMBOL)


# -----------------------
# Binance helpers
# -----------------------
def resync_time(client: Client) -> int:
    server_ts = client.get_server_time()["serverTime"]
    local_ts = int(time.time() * 1000)
    client.timestamp_offset = server_ts - local_ts
    return client.timestamp_offset


def safe_call(client: Client, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except BinanceAPIException as e:
        if "code=-1021" in str(e):
            resync_time(client)
            return fn(*args, **kwargs)
        raise


def quantize_down(value, step) -> Decimal:
    v = Decimal(str(value))
    s = Decimal(str(step))
    return (v / s).to_integral_value(rounding=ROUND_DOWN) * s


def dec_to_str(d: Decimal) -> str:
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def get_symbol_filters(client: Client, symbol: str) -> Dict[str, Any]:
    info = safe_call(client, client.get_symbol_info, symbol)
    return {f["filterType"]: f for f in info["filters"]}


def get_asset_balance(client: Client, asset: str) -> tuple[float, float, float]:
    acct = safe_call(client, client.get_account)

    for bal in acct["balances"]:
        if bal["asset"] == asset:
            free = float(bal["free"])
            locked = float(bal["locked"])
            return free, locked, free + locked

    return 0.0, 0.0, 0.0


def group_open_sell_orders(open_orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Groups open SELL orders by orderListId.

    For OCO TP/SL, Binance returns two SELL orders with the same orderListId.
    We should treat those as one protected trade and sell only one quantity,
    not the sum of both legs.
    """
    groups: Dict[str, Dict[str, Any]] = {}

    for order in open_orders:
        if str(order.get("side", "")).upper() != "SELL":
            continue

        order_list_id = order.get("orderListId", -1)
        group_key = str(order_list_id if int(order_list_id) != -1 else order.get("orderId"))

        order_time = int(order.get("time") or order.get("workingTime") or order.get("updateTime") or 0)
        orig_qty = float(order.get("origQty", 0.0))
        executed_qty = float(order.get("executedQty", 0.0))
        remaining_qty = max(0.0, orig_qty - executed_qty)

        existing = groups.get(group_key)
        if existing is None:
            groups[group_key] = {
                "group_key": group_key,
                "order_list_id": int(order_list_id),
                "symbol": order.get("symbol", SYMBOL),
                "first_order_id": int(order["orderId"]),
                "oldest_order_time_ms": order_time,
                "remaining_qty": remaining_qty,
                "orders": [order],
            }
        else:
            existing["oldest_order_time_ms"] = min(existing["oldest_order_time_ms"], order_time)
            existing["remaining_qty"] = max(existing["remaining_qty"], remaining_qty)
            existing["orders"].append(order)

    return sorted(groups.values(), key=lambda g: g["oldest_order_time_ms"])


def cancel_order_group(client: Client, group: Dict[str, Any]) -> Any:
    """
    Cancel OCO by orderListId when possible.
    Fallback: cancel one child order; Binance should cancel the related OCO list.
    """
    symbol = group["symbol"]
    order_list_id = int(group.get("order_list_id", -1))

    if order_list_id != -1:
        params = {"symbol": symbol, "orderListId": order_list_id}

        # Newer python-binance versions expose this auto-generated endpoint.
        if hasattr(client, "v3_delete_order_list"):
            return safe_call(client, client.v3_delete_order_list, **params)

        # Your api_liveScript.py already uses low-level _post for OCO compatibility,
        # so this mirrors that style for DELETE /api/v3/orderList.
        if hasattr(client, "_delete"):
            return safe_call(client, client._delete, "orderList", True, data=params)

    return safe_call(
        client,
        client.cancel_order,
        symbol=symbol,
        orderId=int(group["first_order_id"]),
    )


def market_sell_qty(client: Client, symbol: str, qty: float) -> Dict[str, Any]:
    filters = get_symbol_filters(client, symbol)
    lot = filters["LOT_SIZE"]

    step_size = lot["stepSize"]
    min_qty = float(lot["minQty"])

    qty_dec = quantize_down(qty, step_size)
    qty_final = float(qty_dec)

    if qty_final < min_qty:
        raise ValueError(f"Sell qty {qty_final} is below minQty {min_qty}")

    qty_str = dec_to_str(qty_dec)

    return safe_call(
        client,
        client.order_market_sell,
        symbol=symbol,
        quantity=qty_str,
        newOrderRespType="FULL",
    )


def get_last_price(client: Client, symbol: str) -> float:
    ticker = safe_call(client, client.get_symbol_ticker, symbol=symbol)
    return float(ticker["price"])


def get_min_notional(filters: Dict[str, Any]) -> float:
    """
    Binance may expose min notional via MIN_NOTIONAL or NOTIONAL depending
    on the symbol/API version.
    """
    if "MIN_NOTIONAL" in filters:
        return float(filters["MIN_NOTIONAL"].get("minNotional", 0.0))

    if "NOTIONAL" in filters:
        return float(filters["NOTIONAL"].get("minNotional", 0.0))

    return 0.0


def sweep_free_base_dust(client: Client) -> bool:
    """
    Sell free leftover BASE_ASSET back into quote asset.

    This only sells free BTC, not locked BTC. Locked BTC remains protected by
    active OCO orders.
    """
    if not SWEEP_FREE_BASE_DUST:
        print("Dust sweep disabled.")
        return False

    base_free, base_locked, base_total = get_asset_balance(client, BASE_ASSET)
    sweep_qty_raw = max(0.0, base_free - float(DUST_SWEEP_RESERVED_QTY))

    print(
        f"Dust sweep check: {BASE_ASSET} "
        f"free={base_free:.8f}, locked={base_locked:.8f}, "
        f"total={base_total:.8f}, sweep_qty_raw={sweep_qty_raw:.8f}"
    )

    if sweep_qty_raw <= 0:
        print("Dust sweep: nothing free to sell.")
        return False

    filters = get_symbol_filters(client, SYMBOL)
    lot = filters["LOT_SIZE"]

    step_size = lot["stepSize"]
    min_qty = float(lot["minQty"])
    min_notional = get_min_notional(filters)

    qty_dec = quantize_down(sweep_qty_raw, step_size)
    qty = float(qty_dec)

    if qty < min_qty:
        print(f"Dust sweep skipped: qty={qty:.8f} below minQty={min_qty:.8f}")
        return False

    last_price = get_last_price(client, SYMBOL)
    notional = qty * last_price

    if min_notional > 0 and notional < min_notional:
        print(
            f"Dust sweep skipped: notional={notional:.4f} "
            f"below minNotional={min_notional:.4f}"
        )
        return False

    if not ENABLE_LIVE_TRADING:
        print(f"DRY RUN: would dust-sell {qty:.8f} {BASE_ASSET}")
        return False

    print(f"Dust sweep: market selling {qty:.8f} {BASE_ASSET} on {SYMBOL}")
    sell_resp = market_sell_qty(client, SYMBOL, qty)
    print(f"Dust sweep sell response: {sell_resp}")

    return True


def main() -> int:
    if not API_KEY or not API_SECRET:
        print("ERROR: Missing BINANCE_LIVE_API_KEY / BINANCE_LIVE_API_SECRET", file=sys.stderr)
        return 2

    client = Client(API_KEY, API_SECRET)
    client.API_URL = "https://api.binance.com/api"

    resync_time(client)

    now_ms = int(time.time() * 1000)
    threshold_seconds = HORIZON_EXIT_MINUTES * 60
    effective_threshold_seconds = max(0, threshold_seconds - HORIZON_EXIT_GRACE_SECONDS)

    print("Horizon exit manager")
    print(f"  bot_id={BOT_ID}")
    print(f"  symbol={SYMBOL}")
    print(f"  base_asset={BASE_ASSET}")
    print(f"  resample_rule={RESAMPLE_RULE}")
    print(f"  horizon_steps={HORIZON_STEPS}")
    print(f"  horizon_exit_minutes={HORIZON_EXIT_MINUTES}")
    print(f"  grace_seconds={HORIZON_EXIT_GRACE_SECONDS}")
    print(f"  enable_live_trading={int(ENABLE_LIVE_TRADING)}")
    print(f"  sweep_free_base_dust={int(SWEEP_FREE_BASE_DUST)}")
    print(f"  dust_sweep_reserved_qty={DUST_SWEEP_RESERVED_QTY}")
    print(f"  utc_now={datetime.now(timezone.utc).isoformat()}")

    open_orders = safe_call(client, client.get_open_orders, symbol=SYMBOL)
    groups = group_open_sell_orders(open_orders)

    closed_count = 0

    if not groups:
        print(f"No open SELL protection orders found for {SYMBOL}. Nothing to close.")

    for group in groups:
        order_age_seconds = max(0, (now_ms - int(group["oldest_order_time_ms"])) / 1000)
        order_age_minutes = order_age_seconds / 60

        print(
            f"Group {group['group_key']}: "
            f"orders={len(group['orders'])}, "
            f"remaining_qty={group['remaining_qty']:.8f}, "
            f"age_minutes={order_age_minutes:.2f}"
        )

        if order_age_seconds < effective_threshold_seconds:
            print("  Not old enough. Skipping.")
            continue

        target_qty = float(group["remaining_qty"])
        if target_qty <= 0:
            print("  Remaining qty is zero. Skipping.")
            continue

        if not ENABLE_LIVE_TRADING:
            print("  DRY RUN: would cancel protection order(s), then market sell.")
            continue

        print("  Eligible for horizon exit. Canceling TP/SL protection...")
        cancel_resp = cancel_order_group(client, group)
        print(f"  Cancel response: {cancel_resp}")

        # Give Binance a moment to release locked BTC from the canceled OCO.
        time.sleep(2)

        base_free, base_locked, base_total = get_asset_balance(client, BASE_ASSET)
        sell_qty = min(target_qty, base_free)

        print(
            f"  {BASE_ASSET} balance after cancel: "
            f"free={base_free:.8f}, locked={base_locked:.8f}, total={base_total:.8f}"
        )
        print(f"  Market selling qty={sell_qty:.8f}")

        if sell_qty <= 0:
            print("  No free balance available after cancel. Skipping sell.")
            continue

        sell_resp = market_sell_qty(client, SYMBOL, sell_qty)
        print(f"  Sell response: {sell_resp}")

        closed_count += 1

    print(f"Done. Horizon exits executed: {closed_count}")

    swept = sweep_free_base_dust(client)
    print(f"Dust sweep executed: {int(swept)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
