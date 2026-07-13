"""
features.py — shared OHLCV feature engineering for simulation and live trading.

Created: 2025-06-25
Tag: Standard - Same Equity Curve
      Search the repo for "Standard - Same Equity Curve" to find all sim ↔ live parity edits.

Purpose: single source of truth so export_simulation_equity_curve.py and
api_liveScript.py (and api_testScript.py) compute identical model inputs.

Both paths must import add_features() / build_features() from this module.
Do not duplicate indicator logic in individual scripts.

---------------------------------------------------------------------------
Standard - Same Equity Curve — REMAINING PARITY WORK (as of 2025-06-25)
---------------------------------------------------------------------------
Feature alignment alone does not make backtest PnL match live fills. After
deploying this module, still reconcile the following:

1. MODEL SOURCE (2025-06-25)
   - export_simulation_equity_curve.py retrains XGBClassifier on every run.
   - api_liveScript.py loads a fixed joblib (MODEL_PATH).
   - Fix: export and save the trained model to the same joblib path the live
     bot loads, or load that joblib in the export script instead of retraining.

2. CONFIG DEFAULTS (2025-06-25)
   Align .env with BacktestConfig in export_simulation_equity_curve.py:
   - THRESHOLD=0.4214          (live default was 0.42)
   - TAKE_PROFIT=0.0455        (live default was 0.045)
   - STOP_LOSS=0.0051          (live default was 0.005)
   - PCT_ACCOUNT_PER_TRADE=0.7888  (live default was 0.79)

3. DATA SOURCE / BAR ALIGNMENT (2025-06-25)
   - Simulation resamples 1m CSV to 4h; live uses Binance native 4h klines.
   - Fix: ensure CSV timestamps are UTC and 4h boundaries match Binance
     open_time, or feed Binance klines into the export pipeline for comparison.

4. ENTRY PRICE (2025-06-25)
   - Simulation buys at bar Close; live places a market buy after the candle
     closes (avg fill can differ from Close due to slippage and cron delay).

5. EXIT / TP-SL MODEL (2025-06-25)
   - Simulation checks TP/SL against bar Close only (ignores intrabar High/Low).
   - Live uses Binance OCO orders that trigger intrabar; SL is STOP_LOSS_LIMIT
     with tick-size rounding and a 0.1% limit buffer below the stop.
   - Fix (optional): update run_simulation() to test High/Low per bar for exits.

6. FEES AND SIZING (2025-06-25)
   - Simulation models commission_pct=0.001 on buy and sell before sizing.
   - Live uses Binance fee schedule and LOT_SIZE / PRICE_FILTER quantization.
   - Simulation sizes from equity_cash; live sizes from quote_free balance.

7. RETRAIN AFTER ADX FIX (2025-06-25)
   - Retrain and redeploy using this module so training and inference both use
     Wilder-smoothed ADX_14. Production live model: Models/BTCUSDT4h1307.joblib
     (set MODEL_PATH in .env).
---------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard - Same Equity Curve (2025-06-25): grep this tag across the repo.
STANDARD_SAME_EQUITY_CURVE = "Standard - Same Equity Curve"

FEATURE_COLS = [
    "Volume",
    "returns",
    "log_returns",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "PROC_HORIZON",
    "hour",
    "ADX_14",
]


def add_features(df_bars: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    """
    Compute classifier features from OHLCV bars (any timeframe).

    Standard - Same Equity Curve (2025-06-25): Wilder ADX + notebook indicators;
    must match V2_Bot_BTC.ipynb, export_simulation_equity_curve.py, api_liveScript.py.

    Expects columns: Open, High, Low, Close, Volume.
    Extra columns (e.g. close_time from Binance) are preserved in the output.
    Rows with NaN in any FEATURE_COLS entry are dropped.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df_bars.columns)
    if missing:
        raise ValueError(f"OHLCV dataframe missing columns: {sorted(missing)}")

    df = df_bars.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Time features (index assumed UTC for Binance klines and aligned CSV bars)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Basic returns
    df["returns"] = close.pct_change()
    df["log_returns"] = np.log(close / close.shift(1))
    df["cum_log_returns"] = df["log_returns"].cumsum()

    # RSI(14)
    window_rsi = 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window_rsi).mean()
    avg_loss = loss.rolling(window_rsi).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # EMA / MACD
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    df["EMA_8"] = close.ewm(span=8, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    # Past percentage change over horizon_steps bars (no future data required).
    df["PROC_HORIZON"] = close.pct_change(periods=horizon_steps)

    # ADX(14) — Wilder-style smoothing (notebook / simulation definition).
    window_adx = 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr_smooth = tr.ewm(alpha=1 / window_adx, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / window_adx, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / window_adx, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))

    df["ADX_14"] = dx.ewm(alpha=1 / window_adx, adjust=False).mean()
    df["PLUS_DI_14"] = plus_di
    df["MINUS_DI_14"] = minus_di

    df_features = df.dropna(subset=FEATURE_COLS).copy()
    if df_features.empty:
        raise RuntimeError("No feature-ready rows after indicator calculation.")

    return df_features


# Alias kept for live/test scripts that historically called build_features().
build_features = add_features
