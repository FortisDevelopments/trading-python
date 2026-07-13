#!/usr/bin/env python3
"""
api_testScript.py (cron-ready)

One cycle:
- Load env (.env)
- Healthcheck Binance testnet
- Load trained classifier
- Fetch latest MAINNET 4h klines (for feature history)
- Build features
- Compute live buy probability + signal
- If signal==1 and under MAX_OPEN_TRADES:
    - Place MARKET BUY on TESTNET
    - Attach OCO TP/SL best-effort (non-fatal if unsupported)
- Log run + optional order to your Node/Express API (MySQL)
- (Optional) Also append a local CSV log for backup

Notes:
- Candles are fetched from mainnet (public) for sufficient history, while orders are placed on testnet.
- OCO may be unsupported/limited on some testnet environments. If OCO fails, the script will continue.
- Feature engineering (2025-06-25): shared via features.py.
  Tag: Standard - Same Equity Curve — see features.py for parity checklist.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional

import pandas as pd
import joblib
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

from features import FEATURE_COLS, build_features

# Standard - Same Equity Curve (2025-06-25): same FEATURE_COLS as live + export + notebook.


# Ensure imports from the same folder work under cron (logger_api.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Load .env early so module-level config picks it up
load_dotenv()

# Try to import API logger helpers (optional)
try:
    from logger_api import log_run_and_optional_order
except Exception:
    log_run_and_optional_order = None  # type: ignore


# ==========================
# CONFIG (edit via .env)
# ==========================
BOT_ID = os.getenv("BOT_ID", "btc_4h_testnet3")

# If 1, send logs to backend via logger_api; if logger_api missing, it will be skipped.
ENABLE_API_LOGGING = os.getenv("ENABLE_API_LOGGING", "1") == "1"

# Local CSV backup logging (optional)
ENABLE_CSV_LOGGING = os.getenv("ENABLE_CSV_LOGGING", "1") == "1"
LOG_PATH = os.getenv("TRADE_LOG_PATH", "trade_log.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "Models/BTCUSDT4h1307.joblib")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = Client.KLINE_INTERVAL_4HOUR

THRESHOLD = float(os.getenv("THRESHOLD", "0.31"))
HORIZON_STEPS = int(os.getenv("HORIZON_STEPS", "6"))

TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.070650"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.007443"))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "4"))
PCT_ACCOUNT_PER_TRADE = float(os.getenv("PCT_ACCOUNT_PER_TRADE", "0.05"))

# Standard - Same Equity Curve (2025-06-25): testnet trading params are separate from live/sim defaults.


# ==========================
# Local CSV logging helpers
# ==========================
def append_log(rows: List[dict]) -> None:
    """Append rows to local CSV log (best-effort)."""
    if not ENABLE_CSV_LOGGING or not rows:
        return
    try:
        df = pd.DataFrame(rows)
        header = not os.path.exists(LOG_PATH)
        df.to_csv(LOG_PATH, mode="a", header=header, index=False)
    except Exception:
        # Never crash trading because logging failed
        pass


# ==========================
# API logging helpers
# ==========================
def log_run_and_order_to_api(run_payload: Dict[str, Any], order_payload: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Best-effort DB logging via your Node/Express API.
    Returns run_id if logged successfully, else None.
    """
    if not ENABLE_API_LOGGING or log_run_and_optional_order is None:
        return None
    try:
        resp = log_run_and_optional_order(run_payload, order_payload)
        return resp.get("run_id")
    except Exception as e:
        append_log([{
            "event": "api_log_error",
            "ts": pd.Timestamp.utcnow().isoformat(),
            "message": str(e),
        }])
        return None


# ==========================
# Binance helpers
# ==========================
def resync_time(client_testnet: Client) -> int:
    """Fix APIError -1021 by syncing local clock vs server time."""
    server_ts = client_testnet.get_server_time()["serverTime"]  # ms
    local_ts = int(time.time() * 1000)                          # ms
    client_testnet.timestamp_offset = server_ts - local_ts
    return client_testnet.timestamp_offset


def safe_call(client_testnet: Client, fn, *args, **kwargs):
    """Retry once on -1021 timestamp issues."""
    try:
        return fn(*args, **kwargs)
    except BinanceAPIException as e:
        if "code=-1021" in str(e):
            resync_time(client_testnet)
            return fn(*args, **kwargs)
        raise


def quantize_down(value, step) -> Decimal:
    """Round DOWN to nearest multiple of step using Decimal."""
    v = Decimal(str(value))
    s = Decimal(str(step))
    return (v / s).to_integral_value(rounding=ROUND_DOWN) * s


def dec_to_str(d: Decimal) -> str:
    """Decimal to clean string without scientific notation/trailing zeros."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def get_symbol_filters(client_testnet: Client, symbol: str) -> Dict[str, Any]:
    info = safe_call(client_testnet, client_testnet.get_symbol_info, symbol)
    return {f["filterType"]: f for f in info["filters"]}


def get_free_balance(client_testnet: Client, asset: str) -> tuple[float, float]:
    acct = safe_call(client_testnet, client_testnet.get_account)
    for b in acct["balances"]:
        if b["asset"] == asset:
            return float(b["free"]), float(b["locked"])
    return 0.0, 0.0


def approx_open_trades(client_testnet: Client, symbol: str) -> int:
    """Best-effort open trade count via open OCO, else open orders heuristic."""
    try:
        oco_open = safe_call(client_testnet, client_testnet.get_open_oco_orders)
        return len(oco_open)
    except Exception:
        orders = safe_call(client_testnet, client_testnet.get_open_orders, symbol=symbol)
        return len(orders) // 2


# ==========================
# Market data (MAINNET)
# ==========================
def fetch_4h_klines_mainnet(client_market: Client, symbol: str, limit: int = 1500) -> pd.DataFrame:
    klines = client_market.get_klines(symbol=symbol, interval=INTERVAL, limit=limit)

    cols = [
        "open_time", "Open", "High", "Low", "Close", "Volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(klines, columns=cols)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.set_index("open_time")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = df[c].astype(float)

    df = df[["Open", "High", "Low", "Close", "Volume", "close_time"]].sort_index()

    # keep only closed candles
    now_utc = datetime.now(timezone.utc)
    df = df[df["close_time"] <= now_utc]
    return df


def get_live_signal(
    clf,
    client_market: Client,
    symbol: str,
    threshold: float,
    horizon_steps: int,
    debug: bool = True,
) -> Dict[str, Any]:
    df_4h = fetch_4h_klines_mainnet(client_market, symbol=symbol, limit=1500)
    df_feats = build_features(df_4h, horizon_steps=horizon_steps)
    if df_feats.empty:
        raise RuntimeError("No usable rows after feature engineering. Increase limit.")

    latest = df_feats.iloc[-1]
    X_live = latest[FEATURE_COLS].values.reshape(1, -1)

    p_buy = float(clf.predict_proba(X_live)[:, 1][0])
    signal = int(p_buy >= threshold)

    if debug:
        print("Features timestamp (UTC):", df_feats.index[-1])
        print("Close:", float(latest["Close"]))
        print("P(buy):", p_buy)
        print("Signal:", signal)

    return {
        "timestamp": df_feats.index[-1],
        "close": float(latest["Close"]),
        "p_buy": p_buy,
        "signal": signal,
    }


# ==========================
# Trading (TESTNET)
# ==========================
def market_buy_usdt(client_testnet: Client, symbol: str, usdt_amount: float) -> Dict[str, Any]:
    filters = get_symbol_filters(client_testnet, symbol)
    lot = filters["LOT_SIZE"]
    step_size = lot["stepSize"]
    min_qty = float(lot["minQty"])

    ticker = safe_call(client_testnet, client_testnet.get_symbol_ticker, symbol=symbol)
    price = float(ticker["price"])

    raw_qty = usdt_amount / price
    qty_dec = quantize_down(raw_qty, step_size)
    qty = float(qty_dec)

    if qty < min_qty:
        raise ValueError(f"Calculated qty {qty} < minQty {min_qty}. Increase usdt_amount.")

    qty_str = dec_to_str(qty_dec)
    order = safe_call(client_testnet, client_testnet.order_market_buy, symbol=symbol, quantity=qty_str)

    executed_qty = float(order.get("executedQty", 0.0))
    cumm_quote = float(order.get("cummulativeQuoteQty", 0.0))
    avg_price = (cumm_quote / executed_qty) if executed_qty > 0 else None

    return {"order": order, "executed_qty": executed_qty, "avg_price": avg_price, "qty_str": qty_str}


def place_oco_tp_sl(
    client_testnet: Client,
    symbol: str,
    qty: float,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    sl_limit_buffer_pct: float = 0.001,
) -> Dict[str, Any]:
    """
    Best-effort OCO placement.
    Returns schema + raw response.
    Raises BinanceAPIException / RuntimeError on failure.
    """
    filters = get_symbol_filters(client_testnet, symbol)

    lot = filters["LOT_SIZE"]
    step_size = lot["stepSize"]
    min_qty = float(lot["minQty"])

    price_filter = filters["PRICE_FILTER"]
    tick_size = price_filter["tickSize"]

    qty_dec = quantize_down(qty, step_size)
    if float(qty_dec) < min_qty:
        raise ValueError("qty below minQty")
    qty_str = dec_to_str(qty_dec)

    tp_raw = entry_price * (1 + float(take_profit_pct))
    sl_stop_raw = entry_price * (1 - float(stop_loss_pct))
    sl_limit_raw = sl_stop_raw * (1 - float(sl_limit_buffer_pct))

    tp_dec = quantize_down(tp_raw, tick_size)
    sl_stop_dec = quantize_down(sl_stop_raw, tick_size)
    sl_limit_dec = quantize_down(sl_limit_raw, tick_size)

    tp_str = dec_to_str(tp_dec)
    sl_stop_str = dec_to_str(sl_stop_dec)
    sl_limit_str = dec_to_str(sl_limit_dec)

    print("OCO params:")
    print("  qty      :", qty_str)
    print("  TP price :", tp_str)
    print("  SL stop  :", sl_stop_str)
    print("  SL limit :", sl_limit_str)

    # Try old schema first
    try:
        oco = safe_call(
            client_testnet,
            client_testnet.create_oco_order,
            symbol=symbol,
            side="SELL",
            quantity=qty_str,
            price=tp_str,
            stopPrice=sl_stop_str,
            stopLimitPrice=sl_limit_str,
            stopLimitTimeInForce="GTC",
        )
        return {"schema": "old", "oco": oco, "tp_price": float(tp_dec), "sl_stop": float(sl_stop_dec), "sl_limit": float(sl_limit_dec)}
    except BinanceAPIException:
        pass

    # New schema (may still be unsupported on some testnet environments)
    params = {
        "symbol": symbol,
        "side": "SELL",
        "quantity": qty_str,
        "aboveType": "LIMIT_MAKER",
        "abovePrice": tp_str,
        "belowType": "STOP_LOSS_LIMIT",
        "belowStopPrice": sl_stop_str,
        "belowPrice": sl_limit_str,
        "belowTimeInForce": "GTC",
    }

    if hasattr(client_testnet, "_post"):
        oco = safe_call(client_testnet, client_testnet._post, "orderList/oco", True, data=params)
        return {"schema": "new(_post)", "oco": oco, "tp_price": float(tp_dec), "sl_stop": float(sl_stop_dec), "sl_limit": float(sl_limit_dec)}
    if hasattr(client_testnet, "_request"):
        oco = safe_call(client_testnet, client_testnet._request, "post", "orderList/oco", True, data=params)
        return {"schema": "new(_request)", "oco": oco, "tp_price": float(tp_dec), "sl_stop": float(sl_stop_dec), "sl_limit": float(sl_limit_dec)}

    raise RuntimeError("Client cannot send new OCO schema; upgrade client or use official connector.")


def trade_once_if_signal(
    clf,
    client_market: Client,
    client_testnet: Client,
    symbol: str,
) -> Dict[str, Any]:
    """
    Runs one decision cycle and logs to API (best-effort).
    Returns a response dict with 'action' and details.
    """
    resync_time(client_testnet)

    run_ts = datetime.now(timezone.utc)

    sig = get_live_signal(
        clf,
        client_market=client_market,
        symbol=symbol,
        threshold=THRESHOLD,
        horizon_steps=HORIZON_STEPS,
        debug=True,
    )

    candle_ts = sig["timestamp"].to_pydatetime() if hasattr(sig["timestamp"], "to_pydatetime") else sig["timestamp"]

    # Always fetch balance for logging/decision
    usdt_free, _ = get_free_balance(client_testnet, "USDT")

    # Default run log
    decision = "none"
    message = None
    order_payload = None

    if sig["signal"] != 1:
        print("No BUY signal. No action.")
        decision = "none"

    else:
        open_trades = approx_open_trades(client_testnet, symbol)
        print(f"Open trades (approx): {open_trades} / MAX_OPEN_TRADES: {MAX_OPEN_TRADES}")

        if open_trades >= int(MAX_OPEN_TRADES):
            print("Max open trades reached. No new trade.")
            decision = "skipped_max_open_trades"
        else:
            usdt_amount = usdt_free * float(PCT_ACCOUNT_PER_TRADE)
            if usdt_amount < 10:
                print(f"USDT free={usdt_free:.2f}. Computed trade amount={usdt_amount:.2f} too small.")
                decision = "skipped_small_notional"
            else:
                print(f"USDT free={usdt_free:.2f}. Using usdt_amount={usdt_amount:.2f} (pct={PCT_ACCOUNT_PER_TRADE})")

                buy = market_buy_usdt(client_testnet, symbol=symbol, usdt_amount=usdt_amount)
                qty = buy["executed_qty"]
                entry = buy["avg_price"]

                if entry is None or qty <= 0:
                    raise RuntimeError("Buy did not execute properly; cannot place TP/SL.")

                print(f"Bought qty={qty:.8f} @ avg entry={entry:.2f}")

                # Try OCO (non-fatal if it fails)
                oco = None
                tp_price = entry * (1 + TAKE_PROFIT)
                sl_stop = entry * (1 - STOP_LOSS)
                sl_limit = sl_stop * (1 - 0.001)

                try:
                    oco = place_oco_tp_sl(
                        client_testnet,
                        symbol=symbol,
                        qty=qty,
                        entry_price=entry,
                        take_profit_pct=TAKE_PROFIT,
                        stop_loss_pct=STOP_LOSS,
                        sl_limit_buffer_pct=0.001,
                    )
                    # Prefer quantized values from OCO function
                    tp_price = oco.get("tp_price", tp_price)
                    sl_stop = oco.get("sl_stop", sl_stop)
                    sl_limit = oco.get("sl_limit", sl_limit)
                    decision = "bought"
                except Exception as e:
                    # We bought, but failed to attach OCO
                    decision = "bought_oco_failed"
                    message = f"OCO failed: {e}"
                    print("WARNING:", message)

                # Build order payload for DB (raw_json intentionally omitted)
                order_payload = {
                    "bot_id": BOT_ID,
                    "symbol": symbol,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "order_id": buy.get("order", {}).get("orderId"),
                    "status": buy.get("order", {}).get("status"),
                    "qty": qty,
                    "avg_price": entry,
                    "quote_spent": float(buy.get("order", {}).get("cummulativeQuoteQty", 0.0)) if buy.get("order") else None,
                    "tp_price": tp_price,
                    "sl_stop": sl_stop,
                    "sl_limit": sl_limit,
                    "oco_schema": (oco.get("schema") if isinstance(oco, dict) else None),
                    "executed_at": run_ts,
                    "raw_json": None,
                }

    # Build run payload for DB (raw_json not used here)
    run_payload = {
        "bot_id": BOT_ID,
        "run_ts": run_ts,
        "candle_ts": candle_ts,
        "close_price": sig.get("close"),
        "p_buy": sig.get("p_buy"),
        "signal": sig.get("signal"),
        "threshold": THRESHOLD,
        "horizon_steps": HORIZON_STEPS,
        "usdt_free": usdt_free,
        "decision": decision,
        "message": message,
    }

    # Local CSV heartbeat (optional)
    append_log([{
        "event": "run",
        "ts": pd.Timestamp.utcnow().isoformat(),
        "candle_ts": str(sig.get("timestamp")),
        "close": sig.get("close"),
        "p_buy": sig.get("p_buy"),
        "signal": sig.get("signal"),
        "decision": decision,
        "message": message,
    }])

    # DB logging (best-effort)
    run_id = log_run_and_order_to_api(run_payload, order_payload)

    resp = {
        "bot_id": BOT_ID,
        "run_id": run_id,
        "signal": sig,
        "action": decision,
        "message": message,
        "order_payload": order_payload,
    }
    return resp


def log_open_orders(client_testnet: Client, symbol: str) -> None:
    """
    Prints open orders and (optionally) logs them to CSV only.
    (Not stored in DB in the current 2-table design.)
    """
    open_orders = safe_call(client_testnet, client_testnet.get_open_orders, symbol=symbol)
    print("Open orders:", len(open_orders))

    rows = [{
        "event": "open_order",
        "ts": pd.Timestamp.utcnow().isoformat(),
        "symbol": o.get("symbol"),
        "orderId": o.get("orderId"),
        "side": o.get("side"),
        "type": o.get("type"),
        "status": o.get("status"),
        "origQty": o.get("origQty"),
        "executedQty": o.get("executedQty"),
        "price": o.get("price"),
        "stopPrice": o.get("stopPrice"),
    } for o in open_orders]

    append_log(rows)


def main() -> int:
    API_KEY_TESTNET = os.getenv("BINANCE_TESTNET_API_KEY")
    API_SECRET_TESTNET = os.getenv("BINANCE_TESTNET_API_SECRET")

    if not API_KEY_TESTNET or not API_SECRET_TESTNET:
        print("ERROR: Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET in environment/.env", file=sys.stderr)
        return 2

    print("API_KEY loaded:", API_KEY_TESTNET is not None)
    print("API_SECRET loaded:", API_SECRET_TESTNET is not None)

    # Mainnet client for candles (public)
    client_market = Client()

    # Testnet client for orders
    client_testnet = Client(API_KEY_TESTNET, API_SECRET_TESTNET)
    client_testnet.API_URL = "https://testnet.binance.vision/api"
    print("API_URL:", client_testnet.API_URL)

    server_time = client_testnet.get_server_time()
    print("Testnet server time:", server_time)

    # Load model
    model_path = MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(SCRIPT_DIR, model_path)

    clf = joblib.load(model_path)

    # Run one cycle
    try:
        resp = trade_once_if_signal(clf, client_market, client_testnet, symbol=SYMBOL)
        log_open_orders(client_testnet, symbol=SYMBOL)
        print("DONE. action:", resp.get("action"), "run_id:", resp.get("run_id"))
        return 0
    except Exception as e:
        # Best-effort: log an error run (without candle info)
        run_payload = {
            "bot_id": BOT_ID,
            "run_ts": datetime.now(timezone.utc),
            "candle_ts": None,
            "close_price": None,
            "p_buy": None,
            "signal": None,
            "threshold": THRESHOLD,
            "horizon_steps": HORIZON_STEPS,
            "usdt_free": None,
            "decision": "error",
            "message": str(e),
        }
        log_run_and_order_to_api(run_payload, order_payload=None)

        append_log([{
            "event": "error",
            "ts": pd.Timestamp.utcnow().isoformat(),
            "message": str(e),
            "symbol": SYMBOL,
        }])
        print("ERROR:", e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

