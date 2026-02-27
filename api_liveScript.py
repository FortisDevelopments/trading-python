#!/usr/bin/env python3
"""
api_liveScript.py (cron-ready, LIVE)

One cycle:
- Load env (.env)
- Healthcheck Binance LIVE
- Load trained classifier (classification buy-signal model)
- Fetch latest MAINNET klines (default 4h) for feature history
- Build features
- Compute buy probability + signal
- If signal==1 and under MAX_OPEN_TRADES and ENABLE_LIVE_TRADING=1:
    - Place MARKET BUY on LIVE
    - Attach OCO TP/SL (best-effort; logs failure but does not crash)
- Log run + optional order to your Node/Express API (MySQL) via logger_api.py
- (Optional) Also append a local CSV log for backup

Safety:
- By default, ENABLE_LIVE_TRADING=0, so no real orders are placed.
  Set ENABLE_LIVE_TRADING=1 in .env once you are ready.

Config (from .env, with defaults matching your requested live settings):
- MODEL_PATH=btxxx.joblib (default: btc_4h_xgb_classifier5k.joblib)
- RESAMPLE_RULE=4h  (maps to Binance interval; default 4h)
- HORIZON_STEPS=6
- TARGET_SIMPLE_RETURN=0.0035  (logged only)
- THRESHOLD=0.42
- TAKE_PROFIT=0.045
- STOP_LOSS=0.005
- MAX_OPEN_TRADES=3
- PCT_ACCOUNT_PER_TRADE=0.79

Requires:
- python-binance, python-dotenv, requests, joblib, pandas, numpy
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Ensure imports from the same folder work under cron (logger_api.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Load .env early so module-level config picks it up (logger_api reads env at import time)
load_dotenv()

# Try to import API logger helpers (optional)
try:
    from logger_api import log_run_and_optional_order
except Exception:
    log_run_and_optional_order = None  # type: ignore


# ==========================
# CONFIG (edit via .env)
# ==========================
BOT_ID = os.getenv("BOT_ID", "btc_4h_live")

ENABLE_API_LOGGING = os.getenv("ENABLE_API_LOGGING", "1") == "1"
ENABLE_CSV_LOGGING = os.getenv("ENABLE_CSV_LOGGING", "1") == "1"

# Safety switch: set to 1 when ready to place real orders
ENABLE_LIVE_TRADING = os.getenv("ENABLE_LIVE_TRADING", "0") == "1"

LOG_PATH = os.getenv("TRADE_LOG_PATH", "trade_log_live.csv")

MODEL_PATH = os.getenv("MODEL_PATH", "btc_4h_xgb_classifier5k.joblib")

SYMBOL = os.getenv("SYMBOL", "BTCUSDC")
RESAMPLE_RULE = os.getenv("RESAMPLE_RULE", "4h").lower()  # logged; also maps to interval

HORIZON_STEPS = int(os.getenv("HORIZON_STEPS", "6"))
TARGET_SIMPLE_RETURN = float(os.getenv("TARGET_SIMPLE_RETURN", "0.0035"))  # logged only

THRESHOLD = float(os.getenv("THRESHOLD", "0.42"))

TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.045"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.005"))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "3"))
PCT_ACCOUNT_PER_TRADE = float(os.getenv("PCT_ACCOUNT_PER_TRADE", "0.79"))

# Feature columns expected by the model
FEATURE_COLS = [
    "Volume",
    "returns", "log_returns",
    "RSI_14",
    "MACD", "MACD_signal",
    "PROC_HORIZON",
    "hour",
    "ADX_14",
]


# ==========================
# Interval mapping
# ==========================
def rule_to_interval(rule: str) -> str:
    r = rule.strip().lower()
    mapping = {
        "1h": Client.KLINE_INTERVAL_1HOUR,
        "2h": Client.KLINE_INTERVAL_2HOUR,
        "4h": Client.KLINE_INTERVAL_4HOUR,
        "6h": Client.KLINE_INTERVAL_6HOUR,
        "8h": Client.KLINE_INTERVAL_8HOUR,
        "12h": Client.KLINE_INTERVAL_12HOUR,
        "1d": Client.KLINE_INTERVAL_1DAY,
        "3d": Client.KLINE_INTERVAL_3DAY,
        "1w": Client.KLINE_INTERVAL_1WEEK,
        "1m": Client.KLINE_INTERVAL_1MONTH,
    }
    if r not in mapping:
        raise ValueError(f"Unsupported RESAMPLE_RULE '{rule}'. Supported: {sorted(mapping.keys())}")
    return mapping[r]


INTERVAL = rule_to_interval(RESAMPLE_RULE)


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
        pass


# ==========================
# API logging helpers
# ==========================
def log_run_and_order_to_api(run_payload: Dict[str, Any], order_payload: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Best-effort DB logging via your Node/Express API. Returns run_id if ok."""
    if not ENABLE_API_LOGGING or log_run_and_optional_order is None:
        return None
    try:
        resp = log_run_and_optional_order(run_payload, order_payload)
        return resp.get("run_id")
    except Exception as e:
        append_log([{
            "event": "api_log_error",
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "message": str(e),
        }])
        return None


# ==========================
# Binance helpers
# ==========================
def resync_time(client_live: Client) -> int:
    """Fix APIError -1021 by syncing local clock vs server time."""
    server_ts = client_live.get_server_time()["serverTime"]  # ms
    local_ts = int(time.time() * 1000)                       # ms
    client_live.timestamp_offset = server_ts - local_ts
    return client_live.timestamp_offset


def safe_call(client_live: Client, fn, *args, **kwargs):
    """Retry once on -1021 timestamp issues."""
    try:
        return fn(*args, **kwargs)
    except BinanceAPIException as e:
        if "code=-1021" in str(e):
            resync_time(client_live)
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


def infer_quote_asset(symbol: str) -> str:
    s = symbol.upper()
    for q in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD"):
        if s.endswith(q):
            return q
    return os.getenv("QUOTE_ASSET", "USDT").upper()


def get_symbol_filters(client_live: Client, symbol: str) -> Dict[str, Any]:
    info = safe_call(client_live, client_live.get_symbol_info, symbol)
    return {f["filterType"]: f for f in info["filters"]}


def get_free_balance(client_live: Client, asset: str) -> Tuple[float, float]:
    acct = safe_call(client_live, client_live.get_account)
    for b in acct["balances"]:
        if b["asset"] == asset:
            return float(b["free"]), float(b["locked"])
    return 0.0, 0.0


def approx_open_trades(client_live: Client, symbol: str) -> int:
    """Count open positions by counting open OCO order lists; fallback to open orders heuristic."""
    try:
        oco_open = safe_call(client_live, client_live.get_open_oco_orders)
        return len(oco_open)
    except Exception:
        orders = safe_call(client_live, client_live.get_open_orders, symbol=symbol)
        return max(0, len(orders) // 2)


# ==========================
# Market data (MAINNET)
# ==========================
def fetch_klines_mainnet(client_market: Client, symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    klines = client_market.get_klines(symbol=symbol, interval=interval, limit=limit)

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


# ==========================
# Feature engineering
# ==========================
def build_features(df_ohlcv: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    df = df_ohlcv.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["hour"] = df.index.hour  # UTC hour

    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))

    # RSI_14
    window_rsi = 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window_rsi).mean()
    avg_loss = loss.rolling(window_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (12,26,9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # PROC_HORIZON = % change over horizon_steps bars (your definition)
    df["PROC_HORIZON"] = close.pct_change(periods=horizon_steps)

    # ADX_14 (rolling approximation)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100
    df["ADX_14"] = dx.rolling(14).mean()

    df = df.dropna()

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns after build: {missing}")

    return df


def get_live_signal(
    clf,
    client_market: Client,
    symbol: str,
    interval: str,
    threshold: float,
    horizon_steps: int,
    debug: bool = True,
) -> Dict[str, Any]:
    df = fetch_klines_mainnet(client_market, symbol=symbol, interval=interval, limit=1500)
    df_feats = build_features(df, horizon_steps=horizon_steps)
    if df_feats.empty:
        raise RuntimeError("No usable rows after feature engineering. Increase limit.")

    latest = df_feats.iloc[-1]
    X_live = latest[FEATURE_COLS].values.reshape(1, -1)

    p_buy = float(clf.predict_proba(X_live)[:, 1][0])
    signal = int(p_buy >= threshold)

    if debug:
        print("RESAMPLE_RULE:", RESAMPLE_RULE, "INTERVAL:", interval)
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
# Trading (LIVE)
# ==========================
def market_buy_quote_amount(client_live: Client, symbol: str, quote_amount: float) -> Dict[str, Any]:
    """
    Prefer quoteOrderQty for market buys (reduces LOT_SIZE rounding issues).
    Falls back to quantity-based order if quoteOrderQty isn't accepted.
    Returns executed_qty and avg_price.
    """
    # Attempt quoteOrderQty (supported for many spot symbols)
    try:
        order = safe_call(
            client_live,
            client_live.order_market_buy,
            symbol=symbol,
            quoteOrderQty=str(float(quote_amount)),
        )
    except Exception:
        # Fallback to quantity-based market buy with LOT_SIZE rounding
        filters = get_symbol_filters(client_live, symbol)
        lot = filters["LOT_SIZE"]
        step_size = lot["stepSize"]
        min_qty = float(lot["minQty"])

        ticker = safe_call(client_live, client_live.get_symbol_ticker, symbol=symbol)
        price = float(ticker["price"])

        raw_qty = quote_amount / price
        qty_dec = quantize_down(raw_qty, step_size)
        qty = float(qty_dec)

        if qty < min_qty:
            raise ValueError(f"Calculated qty {qty} < minQty {min_qty}. Increase quote_amount.")

        qty_str = dec_to_str(qty_dec)
        order = safe_call(client_live, client_live.order_market_buy, symbol=symbol, quantity=qty_str)

    executed_qty = float(order.get("executedQty", 0.0))
    cumm_quote = float(order.get("cummulativeQuoteQty", 0.0))
    avg_price = (cumm_quote / executed_qty) if executed_qty > 0 else None

    return {"order": order, "executed_qty": executed_qty, "avg_price": avg_price}


def place_oco_tp_sl(
    client_live: Client,
    symbol: str,
    qty: float,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
    sl_limit_buffer_pct: float = 0.001,
) -> Dict[str, Any]:
    """
    Place OCO TP/SL on LIVE.

    Binance Spot OCO recently moved to a "new schema" that requires:
      aboveType / abovePrice (TP leg) and belowType / belowStopPrice / belowPrice (SL leg).

    We send the new schema via the underlying request method for maximum compatibility with
    python-binance versions that don't expose it directly.

    Returns: schema, raw response, and quantized tp/sl prices.
    """
    filters = get_symbol_filters(client_live, symbol)

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

    # New OCO schema params
    params = {
        "symbol": symbol,
        "side": "SELL",
        "quantity": qty_str,

        # TP leg ("above")
        # LIMIT_MAKER is a good TP default: maker-only limit at the TP price.
        "aboveType": "LIMIT_MAKER",
        "abovePrice": tp_str,

        # SL leg ("below")
        "belowType": "STOP_LOSS_LIMIT",
        "belowStopPrice": sl_stop_str,
        "belowPrice": sl_limit_str,
        "belowTimeInForce": "GTC",
    }

    # Some environments expect aboveTimeInForce when aboveType is LIMIT (not LIMIT_MAKER).
    # We keep LIMIT_MAKER to avoid needing it, but you can switch if desired.
    # params["aboveTimeInForce"] = "GTC"

    # Send via underlying client methods (python-binance compatibility)
    def _send(p: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
        if hasattr(client_live, "_post"):
            return safe_call(client_live, client_live._post, "orderList/oco", True, data=p), "new(_post)"
        if hasattr(client_live, "_request"):
            return safe_call(client_live, client_live._request, "post", "orderList/oco", True, data=p), "new(_request)"
        raise RuntimeError("Client cannot send new OCO schema; upgrade python-binance or use official connector.")

    try:
        oco, schema = _send(params)
    except BinanceAPIException as e:
        # Some environments reject LIMIT_MAKER in OCO; retry with LIMIT + GTC if that happens.
        if ("code=-1158" in str(e)) or ("Order type not supported" in str(e)):
            params2 = dict(params)
            params2["aboveType"] = "LIMIT"
            params2["aboveTimeInForce"] = "GTC"
            oco, schema = _send(params2)
            schema = schema + "+retry_above_LIMIT"
        else:
            raise


    return {
        "schema": schema,
        "oco": oco,
        "tp_price": float(tp_dec),
        "sl_stop": float(sl_stop_dec),
        "sl_limit": float(sl_limit_dec),
    }


def trade_once_if_signal(
    clf,
    client_market: Client,
    client_live: Client,
    symbol: str,
    interval: str,
) -> Dict[str, Any]:
    resync_time(client_live)
    run_ts = datetime.now(timezone.utc)

    sig = get_live_signal(
        clf,
        client_market=client_market,
        symbol=symbol,
        interval=interval,
        threshold=THRESHOLD,
        horizon_steps=HORIZON_STEPS,
        debug=True,
    )

    candle_ts = sig["timestamp"].to_pydatetime() if hasattr(sig["timestamp"], "to_pydatetime") else sig["timestamp"]
    quote_asset = infer_quote_asset(symbol)

    quote_free, _ = get_free_balance(client_live, quote_asset)

    decision = "none"
    message = None
    order_payload = None

    if sig["signal"] != 1:
        print("No BUY signal. No action.")
        decision = "none"
    else:
        open_trades = approx_open_trades(client_live, symbol)
        print(f"Open trades (approx): {open_trades} / MAX_OPEN_TRADES: {MAX_OPEN_TRADES}")

        if open_trades >= int(MAX_OPEN_TRADES):
            print("Max open trades reached. No new trade.")
            decision = "skipped_max_open_trades"
        else:
            quote_amount = quote_free * float(PCT_ACCOUNT_PER_TRADE)
            if quote_amount <= 0:
                decision = "skipped_no_balance"
                message = f"{quote_asset} free={quote_free:.8f}"
                print(message)
            else:
                print(f"{quote_asset} free={quote_free:.2f}. Using quote_amount={quote_amount:.2f} (pct={PCT_ACCOUNT_PER_TRADE})")

                if not ENABLE_LIVE_TRADING:
                    decision = "dry_run_signal_1"
                    message = "ENABLE_LIVE_TRADING=0 (no order placed)"
                    print("DRY RUN: signal=1 but live trading disabled.")
                else:
                    buy = market_buy_quote_amount(client_live, symbol=symbol, quote_amount=quote_amount)
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
                            client_live,
                            symbol=symbol,
                            qty=qty,
                            entry_price=entry,
                            take_profit_pct=TAKE_PROFIT,
                            stop_loss_pct=STOP_LOSS,
                            sl_limit_buffer_pct=0.001,
                        )
                        tp_price = oco.get("tp_price", tp_price)
                        sl_stop = oco.get("sl_stop", sl_stop)
                        sl_limit = oco.get("sl_limit", sl_limit)
                        decision = "bought"
                    except Exception as e:
                        decision = "bought_oco_failed"
                        message = f"OCO failed: {e}"
                        print("WARNING:", message)

                    order_payload = {
                        "bot_id": BOT_ID,
                        "run_id": None,  # filled by logger helper
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

    run_payload = {
        "bot_id": BOT_ID,
        "run_ts": run_ts,
        "candle_ts": candle_ts,
        "close_price": sig.get("close"),
        "p_buy": sig.get("p_buy"),
        "signal": sig.get("signal"),
        "threshold": THRESHOLD,
        "horizon_steps": HORIZON_STEPS,
        "usdt_free": quote_free,  # stores quote asset free balance (USDT/USDC/etc.)
        "decision": decision,
        "message": message,
        # Extra fields you might later add to backend:
        # "resample_rule": RESAMPLE_RULE,
        # "target_simple_return": TARGET_SIMPLE_RETURN,
        # "quote_asset": infer_quote_asset(symbol),
    }

    append_log([{
        "event": "run",
        "ts": pd.Timestamp.now("UTC").isoformat(),
        "candle_ts": str(sig.get("timestamp")),
        "close": sig.get("close"),
        "p_buy": sig.get("p_buy"),
        "signal": sig.get("signal"),
        "decision": decision,
        "message": message,
        "resample_rule": RESAMPLE_RULE,
        "target_simple_return": TARGET_SIMPLE_RETURN,
    }])

    return {
        "bot_id": BOT_ID,
        "run_id": None,
        "signal": sig,
        "action": decision,
        "message": message,
        "run_payload": run_payload,
        "order_payload": order_payload,
    }


def main() -> int:
    script_start = time.perf_counter()

    API_KEY_LIVE = os.getenv("BINANCE_LIVE_API_KEY")
    API_SECRET_LIVE = os.getenv("BINANCE_LIVE_API_SECRET")

    if not API_KEY_LIVE or not API_SECRET_LIVE:
        print("ERROR: Missing BINANCE_LIVE_API_KEY or BINANCE_LIVE_API_SECRET in environment/.env", file=sys.stderr)
        return 2

    print("LIVE API_KEY loaded:", API_KEY_LIVE is not None)
    print("LIVE API_SECRET loaded:", API_SECRET_LIVE is not None)

    # Public mainnet client for candles
    client_market = Client()

    # Live trading client
    client_live = Client(API_KEY_LIVE, API_SECRET_LIVE)
    # Default API_URL is mainnet; set explicitly for clarity
    client_live.API_URL = "https://api.binance.com/api"
    print("API_URL:", client_live.API_URL)

    # Healthcheck/time sync
    server_time = client_live.get_server_time()
    print("Server time:", server_time)
    print("timestamp_offset set to:", resync_time(client_live))

    # Load model
    model_path = MODEL_PATH
    if not os.path.isabs(model_path):
        model_path = os.path.join(SCRIPT_DIR, model_path)

    clf = joblib.load(model_path)
    print("Loaded model:", os.path.basename(model_path))
    print("Settings:",
          f"RESAMPLE_RULE={RESAMPLE_RULE}",
          f"HORIZON_STEPS={HORIZON_STEPS}",
          f"TARGET_SIMPLE_RETURN={TARGET_SIMPLE_RETURN}",
          f"THRESHOLD={THRESHOLD}",
          f"TP={TAKE_PROFIT}",
          f"SL={STOP_LOSS}",
          f"MAX_OPEN_TRADES={MAX_OPEN_TRADES}",
          f"PCT_PER_TRADE={PCT_ACCOUNT_PER_TRADE}",
          f"ENABLE_LIVE_TRADING={int(ENABLE_LIVE_TRADING)}",
          sep="\n  ")

    try:
        resp = trade_once_if_signal(clf, client_market, client_live, symbol=SYMBOL, interval=INTERVAL)

        # Log full end-to-end script duration (includes setup + signal + order placement if any)
        run_duration_seconds = round(time.perf_counter() - script_start, 3)

        run_payload = dict(resp.get("run_payload") or {})
        run_payload["run_duration_seconds"] = run_duration_seconds

        run_id = log_run_and_order_to_api(run_payload, resp.get("order_payload"))
        resp["run_id"] = run_id

        append_log([{
            "event": "run_duration",
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "run_duration_seconds": run_duration_seconds,
            "action": resp.get("action"),
            "run_id": run_id,
        }])

        print("DONE. action:", resp.get("action"), "run_id:", resp.get("run_id"),
              "run_duration_seconds:", run_duration_seconds)
        return 0
    except Exception as e:
        run_duration_seconds = round(time.perf_counter() - script_start, 3)
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
            "run_duration_seconds": run_duration_seconds,
        }
        log_run_and_order_to_api(run_payload, order_payload=None)

        append_log([{
            "event": "error",
            "ts": pd.Timestamp.now("UTC").isoformat(),
            "message": str(e),
            "symbol": SYMBOL,
        }])
        print("ERROR:", e, "| run_duration_seconds:", run_duration_seconds, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

