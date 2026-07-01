#!/usr/bin/env python3
"""
export_simulation_equity_curve.py

Standalone import -> train -> simulate -> export pipeline for the simulation equity curve.

It is based on your notebook workflow, but it only exports the equity curve CSV:
    datetime,portfolio_value_usd

Default usage:
    python export_simulation_equity_curve.py \
        --data-path complete_dataN.csv \
        --output-path classification/simulation_equity_curve.csv

Also exports simulation runs/orders CSVs (API-key columns aligned with live bot logging):
    simulation_runs.csv   — one row per bar (runs.* fields)
    simulation_orders.csv — one row per simulated BUY (orders.* fields)

Notes:
- Training labels still require future bars, so the training dataframe drops the last
  HORIZON_STEPS rows.
- Simulation predictions use feature-ready rows, not label-ready rows. That means the
  exported equity curve can extend to the latest available resampled candle instead of
  stopping HORIZON_STEPS bars early.
- Feature engineering is shared via features.py (2025-06-25).
  Tag: Standard - Same Equity Curve — search repo for that string.
  See features.py for the full simulation-vs-live parity checklist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import FEATURE_COLS, add_features

# Standard - Same Equity Curve (2025-06-25): shared FEATURE_COLS with live bot + notebook.


@dataclass(frozen=True)
class BacktestConfig:
    # Data/export
    data_path: str = "/home/full-dataset-fetch/data/btcusdt_1m_master.csv"
    output_path: str = "/home/full-dataset-fetch/data/scheduled/simulation_equity_curve.csv"
    runs_output_path: Optional[str] = None
    orders_output_path: Optional[str] = None

    # Live-parity identity (matches api_liveScript.py /api/bot/* payloads)
    bot_id: str = "sim_btc_4h"
    symbol: str = "BTCUSDT"

    # Model / label knobs
    resample_rule: str = "4h"
    horizon_steps: int = 6
    target_simple_return: float = 0.003472
    threshold: float = 0.4214

    # Simulator knobs
    initial_equity: float = 150.0
    take_profit_pct: float = 0.0455
    stop_loss_pct: float = 0.0051
    max_open_trades: int = 3
    pct_account_per_trade: float = 0.7888
    commission_pct: float = 0.001

    # Train/test split by dates. Use None for earliest/latest available.
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = "2021-11-08"
    test_start_date: Optional[str] = "2026-04-19"
    test_end_date: Optional[str] = None

    # XGBoost knobs from the notebook
    n_estimators: int = 400
    max_depth: int = 8
    learning_rate: float = 0.03
    subsample: float = 0.7
    colsample_bytree: float = 0.8
    random_state: int = 42


def parse_optional_path(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"none", "null", "na", "nan"}:
        return None
    return value


def parse_optional_date(value: Optional[str]) -> Optional[str]:
    return parse_optional_path(value)


def resolve_date(value: Optional[str], fallback: pd.Timestamp) -> pd.Timestamp:
    return pd.to_datetime(value) if value is not None else fallback


def load_minute_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    df = pd.read_csv(path)
    required_cols = {"Date", "Time", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["Date"] = df["Date"].astype(str)
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"].astype(str),
        format="%Y%m%d %H:%M:%S",
        errors="raise",
    )

    df = df.sort_values("DateTime").set_index("DateTime")
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    return df


def make_bars(df_min: pd.DataFrame, resample_rule: str) -> pd.DataFrame:
    df_bars = (
        df_min.resample(resample_rule)
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
        .sort_index()
    )

    if df_bars.empty:
        raise RuntimeError("No bars were created after resampling. Check DATA_PATH and RESAMPLE_RULE.")

    return df_bars


def add_training_labels(df_features: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = df_features.copy()

    future_close = df["Close"].shift(-cfg.horizon_steps)
    df["future_ret"] = (future_close - df["Close"]) / df["Close"]

    # Only label rows where the future return is known.
    df = df.dropna(subset=["future_ret"]).copy()
    df["target_buy"] = (df["future_ret"] >= cfg.target_simple_return).astype(int)

    if df.empty:
        raise RuntimeError("No labeled training rows. Check HORIZON_STEPS and data length.")

    return df


def train_classifier(df_labeled: pd.DataFrame, cfg: BacktestConfig) -> XGBClassifier:
    idx = df_labeled.index
    train_start = resolve_date(cfg.train_start_date, idx.min())
    train_end = resolve_date(cfg.train_end_date, idx.max())
    train_mask = (idx >= train_start) & (idx <= train_end)

    X_train = df_labeled.loc[train_mask, FEATURE_COLS]
    y_train = df_labeled.loc[train_mask, "target_buy"].astype(int)

    if X_train.empty:
        raise RuntimeError(
            f"No training rows in selected range: {train_start} -> {train_end}. "
            "Adjust TRAIN_START_DATE/TRAIN_END_DATE."
        )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        objective="binary:logistic",
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    print("=== TRAINING ===")
    print(f"Train period: {train_start} -> {train_end}")
    print(f"Train samples: {len(X_train):,}")
    print(f"Positives: {pos:,} / {len(y_train):,}")
    print(f"scale_pos_weight: {scale_pos_weight:.6f}")
    print()

    return clf


def predict_test_signals(
    clf: XGBClassifier,
    df_features: pd.DataFrame,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    idx = df_features.index
    test_start = resolve_date(cfg.test_start_date, idx.min())
    test_end = resolve_date(cfg.test_end_date, idx.max())
    test_mask = (idx >= test_start) & (idx <= test_end)

    X_test = df_features.loc[test_mask, FEATURE_COLS]
    if X_test.empty:
        raise RuntimeError(
            f"No test rows in selected range: {test_start} -> {test_end}. "
            "Adjust TEST_START_DATE/TEST_END_DATE."
        )

    p_buy = clf.predict_proba(X_test)[:, 1]
    signal = (p_buy >= cfg.threshold).astype(int)

    test_df = df_features.loc[X_test.index, ["Open", "High", "Low", "Close", "Volume"]].copy()
    test_df["p_buy"] = p_buy
    test_df["signal"] = signal

    print("=== TEST / SIMULATION RANGE ===")
    print(f"Test period requested: {test_start} -> {test_end}")
    print(f"Simulation rows: {len(test_df):,}")
    print(f"Predicted buys: {int(test_df['signal'].sum()):,}")
    print(f"Simulation starts: {test_df.index.min()}")
    print(f"Simulation ends:   {test_df.index.max()}")
    print()

    return test_df


def resolve_sibling_output_path(output_path: str, sibling_name: str) -> Path:
    base = Path(output_path)
    return base.with_name(sibling_name)


def run_simulation(
    test_df: pd.DataFrame,
    cfg: BacktestConfig,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    # Standard - Same Equity Curve (2025-06-25): execution model differs from live OCO —
    # entries/exits use bar Close; live uses market fill + intrabar TP/SL (see features.py).
    prices = test_df["Close"].sort_index()
    pred_buy = test_df["signal"].reindex(prices.index).astype(int)
    p_buy_series = test_df["p_buy"].reindex(prices.index)

    equity_cash = float(cfg.initial_equity)
    open_positions: list[dict] = []
    equity_curve: list[float] = []
    equity_index: list[pd.Timestamp] = []
    trades: list[dict] = []
    runs: list[dict] = []
    orders: list[dict] = []
    total_commission = 0.0
    next_run_id = 1
    next_order_id = 1

    for dt, price in prices.items():
        price = float(price)

        # 1) Update existing positions using TP/SL against the current bar close.
        still_open = []
        for pos in open_positions:
            exit_reason = None
            exit_price = None

            if price >= pos["tp_price"]:
                exit_price = pos["tp_price"]
                exit_reason = "TP"
            elif price <= pos["sl_price"]:
                exit_price = pos["sl_price"]
                exit_reason = "SL"

            if exit_reason is not None:
                sell_value = pos["size"] * exit_price
                sell_commission = sell_value * cfg.commission_pct
                net_sell_value = sell_value - sell_commission

                pnl = net_sell_value - pos["cost_basis"]
                ret = net_sell_value / pos["cost_basis"] - 1.0

                equity_cash += net_sell_value
                total_commission += sell_commission

                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": dt,
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "size": pos["size"],
                        "pnl": pnl,
                        "return_pct": ret * 100.0,
                        "reason": exit_reason,
                    }
                )
            else:
                still_open.append(pos)

        open_positions = still_open

        # 2) Decide and optionally open a new position (mirrors live run + order logging).
        signal = int(pred_buy.loc[dt])
        p_buy = float(p_buy_series.loc[dt])
        order_row: dict | None = None
        run_id = next_run_id
        next_run_id += 1
        usdt_free = equity_cash

        if signal != 1:
            decision = "none"
        elif len(open_positions) >= cfg.max_open_trades:
            decision = "skipped_max_open_trades"
        elif equity_cash <= 0:
            decision = "skipped_no_balance"
        else:
            trade_equity = equity_cash * cfg.pct_account_per_trade
            if trade_equity <= 0:
                decision = "skipped_no_balance"
            else:
                buy_commission = trade_equity * cfg.commission_pct
                net_buy_value = trade_equity - buy_commission
                size = net_buy_value / price

                open_positions.append(
                    {
                        "entry_time": dt,
                        "entry_price": price,
                        "size": size,
                        "cost_basis": trade_equity,  # includes buy commission
                        "tp_price": price * (1 + cfg.take_profit_pct),
                        "sl_price": price * (1 - cfg.stop_loss_pct),
                    }
                )

                equity_cash -= trade_equity
                total_commission += buy_commission
                decision = "bought"

                order_row = {
                    "id": next_order_id,
                    "run_id": run_id,
                    "bot_id": cfg.bot_id,
                    "symbol": cfg.symbol,
                    "side": "BUY",
                    "order_type": "MARKET",
                    "status": "FILLED",
                    "qty": size,
                    "avg_price": price,
                    "quote_spent": trade_equity,
                    "executed_at": dt,
                }
                next_order_id += 1

        runs.append(
            {
                "id": run_id,
                "bot_id": cfg.bot_id,
                "run_ts": dt,
                "candle_ts": dt,
                "close_price": price,
                "p_buy": p_buy,
                "signal": signal,
                "threshold": cfg.threshold,
                "horizon_steps": cfg.horizon_steps,
                "decision": decision,
                "usdt_free": usdt_free,
            }
        )
        if order_row is not None:
            orders.append(order_row)

        # 3) Mark-to-market portfolio value at this timestamp.
        open_value = sum(pos["size"] * price for pos in open_positions)
        total_equity = equity_cash + open_value

        equity_curve.append(total_equity)
        equity_index.append(dt)

    # Close remaining positions at final price so final equity includes closing commission.
    if open_positions:
        last_dt = prices.index[-1]
        last_price = float(prices.iloc[-1])

        for pos in open_positions:
            sell_value = pos["size"] * last_price
            sell_commission = sell_value * cfg.commission_pct
            net_sell_value = sell_value - sell_commission

            pnl = net_sell_value - pos["cost_basis"]
            ret = net_sell_value / pos["cost_basis"] - 1.0

            equity_cash += net_sell_value
            total_commission += sell_commission

            trades.append(
                {
                    "entry_time": pos["entry_time"],
                    "exit_time": last_dt,
                    "entry_price": pos["entry_price"],
                    "exit_price": last_price,
                    "size": pos["size"],
                    "pnl": pnl,
                    "return_pct": ret * 100.0,
                    "reason": "EOD",
                }
            )

        # Make the last exported equity point equal to the final realized EOD equity.
        equity_curve[-1] = equity_cash

    equity_series = pd.Series(equity_curve, index=pd.DatetimeIndex(equity_index), name="portfolio_value_usd")
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    runs_df = pd.DataFrame(runs)
    if not runs_df.empty:
        runs_df = runs_df.sort_values("run_ts").reset_index(drop=True)

    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df = orders_df.sort_values("executed_at").reset_index(drop=True)

    final_equity = float(equity_series.iloc[-1])
    total_return_pct = (final_equity / cfg.initial_equity - 1.0) * 100.0

    num_trades = int(len(trades_df))
    wins = int((trades_df["pnl"] > 0).sum()) if num_trades else 0
    losses = int((trades_df["pnl"] < 0).sum()) if num_trades else 0
    win_rate = (wins / num_trades * 100.0) if num_trades else 0.0

    start_date = equity_series.index[0]
    end_date = equity_series.index[-1]
    years = max((end_date - start_date).days / 365.25, 1e-6)
    model_cagr_pct = ((final_equity / cfg.initial_equity) ** (1 / years) - 1.0) * 100.0

    summary = {
        "initial_equity": float(cfg.initial_equity),
        "final_equity": final_equity,
        "total_return_pct": float(total_return_pct),
        "model_cagr_pct": float(model_cagr_pct),
        "num_trades": num_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": float(win_rate),
        "total_commission": float(total_commission),
        "start_date": str(start_date),
        "end_date": str(end_date),
    }

    return equity_series, trades_df, runs_df, orders_df, summary


RUNS_EXPORT_COLS = [
    "id",
    "bot_id",
    "run_ts",
    "candle_ts",
    "close_price",
    "p_buy",
    "signal",
    "threshold",
    "horizon_steps",
    "decision",
    "usdt_free",
]

ORDERS_EXPORT_COLS = [
    "id",
    "run_id",
    "bot_id",
    "symbol",
    "side",
    "order_type",
    "status",
    "qty",
    "avg_price",
    "quote_spent",
    "executed_at",
]


def export_equity_curve(equity_series: pd.Series, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    export_df = equity_series.rename("portfolio_value_usd").rename_axis("datetime").reset_index()
    export_df.to_csv(path, index=False)
    return path


def export_simulation_runs(runs_df: pd.DataFrame, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if runs_df.empty:
        export_df = pd.DataFrame(columns=RUNS_EXPORT_COLS)
    else:
        export_df = runs_df.reindex(columns=RUNS_EXPORT_COLS)

    export_df.to_csv(path, index=False)
    return path


def export_simulation_orders(orders_df: pd.DataFrame, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if orders_df.empty:
        export_df = pd.DataFrame(columns=ORDERS_EXPORT_COLS)
    else:
        export_df = orders_df.reindex(columns=ORDERS_EXPORT_COLS)

    export_df.to_csv(path, index=False)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import data, simulate strategy, and export only the equity curve CSV.")

    parser.add_argument("--data-path", default=BacktestConfig.data_path)
    parser.add_argument("--output-path", default=BacktestConfig.output_path)
    parser.add_argument("--runs-output-path", default=BacktestConfig.runs_output_path)
    parser.add_argument("--orders-output-path", default=BacktestConfig.orders_output_path)
    parser.add_argument("--bot-id", default=BacktestConfig.bot_id)
    parser.add_argument("--symbol", default=BacktestConfig.symbol)

    parser.add_argument("--resample-rule", default=BacktestConfig.resample_rule)
    parser.add_argument("--horizon-steps", type=int, default=BacktestConfig.horizon_steps)
    parser.add_argument("--target-simple-return", type=float, default=BacktestConfig.target_simple_return)
    parser.add_argument("--threshold", type=float, default=BacktestConfig.threshold)

    parser.add_argument("--initial-equity", type=float, default=BacktestConfig.initial_equity)
    parser.add_argument("--take-profit-pct", type=float, default=BacktestConfig.take_profit_pct)
    parser.add_argument("--stop-loss-pct", type=float, default=BacktestConfig.stop_loss_pct)
    parser.add_argument("--max-open-trades", type=int, default=BacktestConfig.max_open_trades)
    parser.add_argument("--pct-account-per-trade", type=float, default=BacktestConfig.pct_account_per_trade)
    parser.add_argument("--commission-pct", type=float, default=BacktestConfig.commission_pct)

    parser.add_argument("--train-start-date", default=BacktestConfig.train_start_date)
    parser.add_argument("--train-end-date", default=BacktestConfig.train_end_date)
    parser.add_argument("--test-start-date", default=BacktestConfig.test_start_date)
    parser.add_argument("--test-end-date", default=BacktestConfig.test_end_date)

    return parser


def config_from_args(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        data_path=args.data_path,
        output_path=args.output_path,
        runs_output_path=parse_optional_path(args.runs_output_path),
        orders_output_path=parse_optional_path(args.orders_output_path),
        bot_id=args.bot_id,
        symbol=args.symbol,
        resample_rule=args.resample_rule,
        horizon_steps=args.horizon_steps,
        target_simple_return=args.target_simple_return,
        threshold=args.threshold,
        initial_equity=args.initial_equity,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_open_trades=args.max_open_trades,
        pct_account_per_trade=args.pct_account_per_trade,
        commission_pct=args.commission_pct,
        train_start_date=parse_optional_date(args.train_start_date),
        train_end_date=parse_optional_date(args.train_end_date),
        test_start_date=parse_optional_date(args.test_start_date),
        test_end_date=parse_optional_date(args.test_end_date),
    )


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    print("=== CONFIG ===")
    for key, value in cfg.__dict__.items():
        print(f"{key}: {value}")
    print()

    df_min = load_minute_data(cfg.data_path)
    df_bars = make_bars(df_min, cfg.resample_rule)
    df_features = add_features(df_bars, cfg.horizon_steps)
    df_labeled = add_training_labels(df_features, cfg)

    print("=== DATA RANGES ===")
    print(f"Raw data:     {df_min.index.min()} -> {df_min.index.max()} ({len(df_min):,} rows)")
    print(f"Bars:         {df_bars.index.min()} -> {df_bars.index.max()} ({len(df_bars):,} rows)")
    print(f"Features:     {df_features.index.min()} -> {df_features.index.max()} ({len(df_features):,} rows)")
    print(f"Train labels: {df_labeled.index.min()} -> {df_labeled.index.max()} ({len(df_labeled):,} rows)")
    print()

    clf = train_classifier(df_labeled, cfg)
    test_df = predict_test_signals(clf, df_features, cfg)
    equity_series, _trades_df, runs_df, orders_df, summary = run_simulation(test_df, cfg)
    output_path = export_equity_curve(equity_series, cfg.output_path)

    runs_output_path = cfg.runs_output_path or str(
        resolve_sibling_output_path(cfg.output_path, "simulation_runs.csv")
    )
    orders_output_path = cfg.orders_output_path or str(
        resolve_sibling_output_path(cfg.output_path, "simulation_orders.csv")
    )
    runs_path = export_simulation_runs(runs_df, runs_output_path)
    orders_path = export_simulation_orders(orders_df, orders_output_path)

    print("=== RESULTS ===")
    print(f"Initial equity:       {summary['initial_equity']:,.2f}")
    print(f"Final equity:         {summary['final_equity']:,.2f}")
    print(f"Total return:         {summary['total_return_pct']:.2f}%")
    print(f"CAGR:                 {summary['model_cagr_pct']:.2f}%")
    print(f"Trades:               {summary['num_trades']}")
    print(f"Wins/Losses/Win rate: {summary['wins']} / {summary['losses']} / {summary['win_rate']:.2f}%")
    print(f"Total commission:     {summary['total_commission']:,.2f}")
    print()
    print(f"Equity curve exported to: {output_path.resolve()}")
    print(f"Runs exported to:         {runs_path.resolve()} ({len(runs_df):,} rows)")
    print(f"Orders exported to:       {orders_path.resolve()} ({len(orders_df):,} rows)")
    print(f"Export range: {equity_series.index.min()} -> {equity_series.index.max()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
