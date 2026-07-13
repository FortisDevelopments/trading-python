#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import timezone

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv("/home/trading-python/.env")

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
API_RETRIES = int(os.getenv("API_RETRIES", "3"))

BOT_ID = os.getenv("BOT_ID", "btc_4h_LIVE")
SYMBOL = os.getenv("SIMULATION_SYMBOL", "BTCUSDT")

CSV_PATH = Path(
    os.getenv(
        "SIMULATION_EQUITY_CSV_PATH",
        "/home/full-dataset-fetch/data/scheduled/simulation_equity_curve.csv",
    )
)


def headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_TOKEN:
        h["Authorization"] = f"Bearer {API_TOKEN}"
    return h


def normalize_ts(value) -> str:
    ts = pd.to_datetime(value)

    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)

    return ts.strftime("%Y-%m-%d %H:%M:%S.000")


def load_points(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Simulation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Support either normal header:
    # datetime,portfolio_value_usd
    # or no-header rows:
    # 2026-05-06 00:00:00,159.85
    if "datetime" not in df.columns or "portfolio_value_usd" not in df.columns:
        df = pd.read_csv(
            csv_path,
            header=None,
            names=["datetime", "portfolio_value_usd"],
        )

    df = df[["datetime", "portfolio_value_usd"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df["portfolio_value_usd"] = pd.to_numeric(
        df["portfolio_value_usd"],
        errors="raise",
    )

    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")

    points = []

    for _, row in df.iterrows():
        points.append(
            {
                "ts": normalize_ts(row["datetime"]),
                "portfolio_value_usd": float(row["portfolio_value_usd"]),
            }
        )

    if not points:
        raise RuntimeError(f"No points loaded from {csv_path}")

    return points


def post_payload(payload: dict) -> dict:
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is not set")

    url = f"{API_BASE_URL}/api/bot/simulation-equity/upload"
    last_error = None

    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers(),
                timeout=API_TIMEOUT,
            )

            if 200 <= response.status_code < 300:
                return response.json() if response.content else {"success": True}

            last_error = RuntimeError(
                f"HTTP {response.status_code}: {response.text[:500]}"
            )
        except Exception as exc:
            last_error = exc

        time.sleep(1.5 * attempt)

    raise last_error


def main() -> int:
    points = load_points(CSV_PATH)

    payload = {
        "bot_id": BOT_ID,
        "symbol": SYMBOL,
        "source_file": str(CSV_PATH),
        "points": points,
    }

    print(f"Uploading simulation equity curve: {CSV_PATH}")
    print(f"Points: {len(points)}")
    print(f"Range: {points[0]['ts']} -> {points[-1]['ts']}")

    response = post_payload(payload)

    print("Upload response:", response)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
