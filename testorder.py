#!/usr/bin/env python3
"""
testorder.py
Tests new logger_api helpers:
  - post_run_id()
  - log_run_and_optional_order()

Env:
  API_BASE_URL=https://api.fortisinvestmentmanagement.com
  API_TOKEN=... (optional)
"""

import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

from logger_api import (
    post_run_id,
    log_run_and_optional_order,
    utc_mysql_datetime_ms,
)


def main() -> int:
    base = os.getenv("API_BASE_URL", "").rstrip("/")
    if not base:
        print("ERROR: API_BASE_URL is not set", file=sys.stderr)
        return 2

    now_utc = datetime.now(timezone.utc)

    # -------------------------
    # 1) Test post_run_id()
    # -------------------------
    run_payload_1 = {
        "bot_id": "btc_4h_testnet3",
        "run_ts": now_utc,  # datetime input; logger_api should normalize
        "candle_ts": now_utc.replace(minute=0, second=0, microsecond=0),
        "close_price": 95002.09,
        "p_buy": 0.55178,
        "signal": 1,
        "threshold": 0.31,
        "horizon_steps": 6,
        "usdt_free": 8916.93,
        "decision": "test_post_run_id",
    }

    print("\n=== Test 1: post_run_id() ===")
    print("POST", f"{base}/api/bot/runs")
    print("Payload:", run_payload_1)

    run_id_1 = post_run_id(run_payload_1)
    print("Returned run_id:", run_id_1)

    # -------------------------
    # 2) Test log_run_and_optional_order() WITH an order
    # -------------------------
    executed_at = utc_mysql_datetime_ms(now_utc)

    run_payload_2 = {
        "bot_id": "btc_4h_testnet3",
        "run_ts": now_utc,
        "candle_ts": now_utc.replace(minute=0, second=0, microsecond=0),
        "close_price": 95555.55,
        "p_buy": 0.60001,
        "signal": 1,
        "threshold": 0.31,
        "horizon_steps": 6,
        "usdt_free": 9000.00,
        "decision": "test_batch_with_order",
    }

    order_payload_2 = {
        "bot_id": "btc_4h_testnet3",
        # run_id will be attached automatically by log_run_and_optional_order()
        "symbol": "BTCUSDT",
        "side": "BUY",
        "order_type": "MARKET",
        "order_id": None,
        "status": "TEST_BATCH",
        "qty": 0.00123,
        "avg_price": 95449.46,
        "quote_spent": 117.44,
        "tp_price": 102192.96,
        "sl_stop": 94739.02,
        "sl_limit": 94644.29,
        "oco_schema": "test_batch",
        "executed_at": executed_at,
        "raw_json": {"note": "batch helper test", "executed_at": executed_at},
    }

    print("\n=== Test 2: log_run_and_optional_order() (with order) ===")
    print("POST", f"{base}/api/bot/runs + /api/bot/orders")
    print("Run payload:", run_payload_2)
    print("Order payload:", order_payload_2)

    batch_resp = log_run_and_optional_order(run_payload_2, order_payload_2)
    print("Batch response:", batch_resp)
    print("Batch run_id:", batch_resp.get("run_id"))

    # -------------------------
    # 3) Test log_run_and_optional_order() WITHOUT an order
    # -------------------------
    run_payload_3 = {
        "bot_id": "btc_4h_testnet3",
        "run_ts": now_utc,
        "candle_ts": now_utc.replace(minute=0, second=0, microsecond=0),
        "close_price": 96000.00,
        "p_buy": 0.12,
        "signal": 0,
        "threshold": 0.31,
        "horizon_steps": 6,
        "usdt_free": 9050.00,
        "decision": "test_batch_no_order",
    }

    print("\n=== Test 3: log_run_and_optional_order() (no order) ===")
    print("POST", f"{base}/api/bot/runs only")
    print("Run payload:", run_payload_3)

    batch_resp_2 = log_run_and_optional_order(run_payload_3, order_payload=None)
    print("Batch response:", batch_resp_2)
    print("Batch run_id:", batch_resp_2.get("run_id"))
    print("Order response should be None:", batch_resp_2.get("order"))

    print("\n✅ All helper tests completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
