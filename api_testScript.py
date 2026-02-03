#!/usr/bin/env python3
"""
Converted from api_testScript.ipynb

Runs one cycle:
- Load env (.env) for Binance testnet keys
- Healthcheck testnet
- Load trained classifier
- Fetch latest MAINNET 4h klines (for feature history)
- Build features
- Compute live buy probability + signal
- If signal==1 and under MAX_OPEN_TRADES, place MARKET BUY on TESTNET and attach OCO TP/SL (best-effort)
- Log signal/trade/spread/open orders to CSV

Notes:
- Candles are fetched from mainnet (public) for sufficient history, while orders are placed on testnet.
- OCO may be unsupported/limited on some testnet environments. If OCO fails, the script will raise.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# ==========================
# CONFIG (edit as needed)
# ==========================
LOG_PATH = os.getenv("TRADE_LOG_PATH", "trade_log.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "btc_4h_xgb_classifier.joblib")

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
INTERVAL = Client.KLINE_INTERVAL_4HOUR

THRESHOLD = float(os.getenv("THRESHOLD", "0.31"))
HORIZON_STEPS = int(os.getenv("HORIZON_STEPS", "6"))

TAKE_PROFIT = float(os.getenv("TAKE_PROFIT", "0.070650"))
STOP_LOSS = float(os.getenv("STOP_LOSS", "0.007443"))

MAX_OPEN_TRADES = int(os.getenv("MAX_OPEN_TRADES", "4"))
PCT_ACCOUNT_PER_TRADE = float(os.getenv("PCT_ACCOUNT_PER_TRADE", "0.05"))

# Feature columns expected by the model
FEATURE_COLS = [
    "Volume",
    "returns", "log_returns",
    "RSI_14",
    "MACD", "MACD_signal",
    "PROC_HORIZON",
    "hour",
    "ADX_14"
]


# ==========================
# Logging helpers
# ==========================
def append_log(rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(LOG_PATH)
    df.to_csv(LOG_PATH, mode="a", header=header, index=False)


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
        "taker_buy_base", "taker_buy_quote", "ignore"
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

    # Sanity: ensure needed columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns after build: {missing}")

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
        return {"schema": "old", "oco": oco}
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
        return {"schema": "new(_post)", "oco": oco}
    if hasattr(client_testnet, "_request"):
        oco = safe_call(client_testnet, client_testnet._request, "post", "orderList/oco", True, data=params)
        return {"schema": "new(_request)", "oco": oco}

    raise RuntimeError("Client cannot send new OCO schema; upgrade client or use official connector.")


def trade_once_if_signal(
    clf,
    client_market: Client,
    client_testnet: Client,
    symbol: str,
) -> Dict[str, Any]:
    resync_time(client_testnet)

    sig = get_live_signal(
        clf,
        client_market=client_market,
        symbol=symbol,
        threshold=THRESHOLD,
        horizon_steps=HORIZON_STEPS,
        debug=True,
    )

    # log signal
    append_log([{
        "event": "signal",
        "ts": pd.Timestamp.utcnow().isoformat(),
        "candle_ts": str(sig["timestamp"]),
        "close": sig["close"],
        "p_buy": sig["p_buy"],
        "signal": sig["signal"],
    }])

    if sig["signal"] != 1:
        print("No BUY signal. No action.")
        return {"signal": sig, "action": "none"}

    open_trades = approx_open_trades(client_testnet, symbol)
    print(f"Open trades (approx): {open_trades} / MAX_OPEN_TRADES: {MAX_OPEN_TRADES}")
    if open_trades >= int(MAX_OPEN_TRADES):
        print("Max open trades reached. No new trade.")
        return {"signal": sig, "action": "skipped_max_open_trades"}

    usdt_free, _ = get_free_balance(client_testnet, "USDT")
    usdt_amount = usdt_free * float(PCT_ACCOUNT_PER_TRADE)
    if usdt_amount < 10:
        print(f"USDT free={usdt_free:.2f}. Computed trade amount={usdt_amount:.2f} too small.")
        return {"signal": sig, "action": "skipped_small_notional"}

    print(f"USDT free={usdt_free:.2f}. Using usdt_amount={usdt_amount:.2f} (pct={PCT_ACCOUNT_PER_TRADE})")

    buy = market_buy_usdt(client_testnet, symbol=symbol, usdt_amount=usdt_amount)
    qty = buy["executed_qty"]
    entry = buy["avg_price"]

    if entry is None or qty <= 0:
        raise RuntimeError("Buy did not execute properly; cannot place OCO.")

    print(f"Bought qty={qty:.8f} @ avg entry={entry:.2f}")

    oco = place_oco_tp_sl(
        client_testnet,
        symbol=symbol,
        qty=qty,
        entry_price=entry,
        take_profit_pct=TAKE_PROFIT,
        stop_loss_pct=STOP_LOSS,
        sl_limit_buffer_pct=0.001,
    )

    resp = {"signal": sig, "action": "bought", "buy": buy, "oco": oco}

    # log trade
    append_log([{
        "event": "trade",
        "ts": pd.Timestamp.utcnow().isoformat(),
        "action": resp.get("action"),
        "p_buy": resp.get("signal", {}).get("p_buy"),
        "candle_ts": str(resp.get("signal", {}).get("timestamp")),
        "buy_orderId": resp.get("buy", {}).get("order", {}).get("orderId"),
        "buy_qty": resp.get("buy", {}).get("executed_qty"),
        "buy_avg_price": resp.get("buy", {}).get("avg_price"),
        "oco_schema": resp.get("oco", {}).get("schema"),
    }])

    # log spread heartbeat
    append_log([{
        "event": "spread",
        "ts": pd.Timestamp.utcnow().isoformat(),
        "candle_ts": str(resp.get("signal", {}).get("timestamp")),
        "signal": resp.get("signal", {}).get("signal"),
        "message": ("BOUGHT" if resp.get("action") == "bought" else "NO_BUY"),
        "signal_close": resp.get("signal", {}).get("close"),
        "fill_avg_price": resp.get("buy", {}).get("avg_price"),
        "spread_abs": ((resp.get("buy", {}).get("avg_price") - resp.get("signal", {}).get("close"))
                       if (resp.get("action") == "bought"
                           and resp.get("buy", {}).get("avg_price") is not None
                           and resp.get("signal", {}).get("close") is not None)
                       else None),
        "spread_pct": ((((resp.get("buy", {}).get("avg_price") / resp.get("signal", {}).get("close")) - 1) * 100)
                       if (resp.get("action") == "bought"
                           and resp.get("buy", {}).get("avg_price") is not None
                           and resp.get("signal", {}).get("close") not in (None, 0))
                       else None),
    }])

    return resp


def log_open_orders(client_testnet: Client, symbol: str) -> None:
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
    load_dotenv()

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
        # resolve relative to script directory
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)

    clf = joblib.load(model_path)

    # Run one cycle
    try:
        resp = trade_once_if_signal(clf, client_market, client_testnet, symbol=SYMBOL)
        log_open_orders(client_testnet, symbol=SYMBOL)
        return 0
    except Exception as e:
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
