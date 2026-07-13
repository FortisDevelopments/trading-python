#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass
class CheckResult:
    level: str
    name: str
    detail: str


class PreflightChecker:
    def __init__(self, root: Path, expected_model: str, expected_symbol: str, expected_quote: str):
        self.root = root
        self.expected_model = expected_model
        self.expected_symbol = expected_symbol.upper()
        self.expected_quote = expected_quote.upper()
        self.results: List[CheckResult] = []
        self.env = self._read_env(root / ".env")

    def add(self, level: str, name: str, detail: str) -> None:
        self.results.append(CheckResult(level, name, detail))

    @staticmethod
    def _read_env(path: Path) -> Dict[str, str]:
        env: Dict[str, str] = {}
        if not path.exists():
            return env
        for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")
        return env

    def check_exists(self, rel_path: str, required: bool = True) -> Path | None:
        p = self.root / rel_path
        if p.exists():
            self.add(PASS, f"file:{rel_path}", f"Found {p}")
            return p
        if required:
            self.add(FAIL, f"file:{rel_path}", f"Missing required file {p}")
        else:
            self.add(WARN, f"file:{rel_path}", f"Optional file not found: {p}")
        return None

    def check_env_value(self, key: str, expected: str | None = None, *, allowed: Tuple[str, ...] | None = None, required: bool = True) -> None:
        value = self.env.get(key)
        if value is None:
            level = FAIL if required else WARN
            self.add(level, f"env:{key}", f"{key} is not set in .env")
            return
        if expected is not None and value != expected:
            self.add(FAIL, f"env:{key}", f"Expected {key}={expected!r}, found {value!r}")
            return
        if allowed is not None and value not in allowed:
            self.add(FAIL, f"env:{key}", f"Expected {key} in {allowed}, found {value!r}")
            return
        self.add(PASS, f"env:{key}", f"{key}={value}")

    def check_env_contains(self, key: str, fragment: str) -> None:
        value = self.env.get(key)
        if value is None:
            self.add(FAIL, f"env:{key}", f"{key} is not set in .env")
            return
        if fragment.lower() not in value.lower():
            self.add(WARN, f"env:{key}", f"{key}={value!r} does not contain {fragment!r}")
            return
        self.add(PASS, f"env:{key}", f"{key} looks consistent with {fragment!r}")

    def check_path_from_env(self, key: str, required: bool = True) -> None:
        value = self.env.get(key)
        if not value:
            level = FAIL if required else WARN
            self.add(level, f"env-path:{key}", f"{key} is not set")
            return
        p = Path(value)
        if not p.is_absolute():
            p = self.root / value
        if p.exists():
            self.add(PASS, f"env-path:{key}", f"{key} points to existing path: {p}")
        else:
            self.add(FAIL, f"env-path:{key}", f"{key} points to missing path: {p}")

    def check_absent(self, rel_path: str) -> None:
        p = self.root / rel_path
        if p.exists():
            self.add(WARN, f"stale:{rel_path}", f"Stale file still present: {p}")
        else:
            self.add(PASS, f"stale:{rel_path}", f"Not present: {p}")

    def check_python_syntax(self, rel_path: str) -> None:
        p = self.root / rel_path
        if not p.exists():
            self.add(FAIL, f"syntax:{rel_path}", "Cannot check syntax because file is missing")
            return
        try:
            ast.parse(p.read_text(encoding="utf-8", errors="ignore"), filename=str(p))
            self.add(PASS, f"syntax:{rel_path}", "Python syntax OK")
        except SyntaxError as e:
            self.add(FAIL, f"syntax:{rel_path}", f"Syntax error at line {e.lineno}: {e.msg}")

    def check_text_contains(self, rel_path: str, patterns: List[Tuple[str, str]], *, must_not_contain: List[Tuple[str, str]] | None = None) -> None:
        p = self.root / rel_path
        if not p.exists():
            self.add(FAIL, f"content:{rel_path}", "Cannot inspect because file is missing")
            return
        text = p.read_text(encoding="utf-8", errors="ignore")
        for needle, label in patterns:
            if needle in text:
                self.add(PASS, f"content:{rel_path}:{label}", f"Found expected text: {needle}")
            else:
                self.add(WARN, f"content:{rel_path}:{label}", f"Did not find expected text: {needle}")
        for needle, label in (must_not_contain or []):
            if needle in text:
                self.add(WARN, f"content:{rel_path}:{label}", f"Found old/reference text still present: {needle}")
            else:
                self.add(PASS, f"content:{rel_path}:{label}", f"Old/reference text not found: {needle}")

    def check_run_shell_loads_env(self) -> None:
        p = self.root / "run_api_live.sh"
        if not p.exists():
            self.add(FAIL, "shell:run_api_live", "run_api_live.sh is missing")
            return
        text = p.read_text(encoding="utf-8", errors="ignore")
        required_snippets = [
            'source "$REPO_DIR/.env"',
            'SCRIPT="$REPO_DIR/api_liveScript.py"',
            'VENV_PY="$REPO_DIR/venv/bin/python"',
        ]
        for snippet in required_snippets:
            if snippet in text:
                self.add(PASS, "shell:run_api_live", f"Found snippet: {snippet}")
            else:
                self.add(WARN, "shell:run_api_live", f"Missing snippet: {snippet}")

    def check_env_consistency(self) -> None:
        symbol = self.env.get("SYMBOL")
        eq = self.env.get("EQUITY_SYMBOL")
        fills = self.env.get("FILLS_SYMBOL")
        if symbol and eq and symbol != eq:
            self.add(FAIL, "env:consistency", f"SYMBOL={symbol!r} != EQUITY_SYMBOL={eq!r}")
        else:
            self.add(PASS, "env:consistency", f"SYMBOL and EQUITY_SYMBOL are aligned ({symbol})")
        if symbol and fills and symbol != fills:
            self.add(FAIL, "env:consistency", f"SYMBOL={symbol!r} != FILLS_SYMBOL={fills!r}")
        else:
            self.add(PASS, "env:consistency", f"SYMBOL and FILLS_SYMBOL are aligned ({symbol})")

    def check_usdt_migration(self) -> None:
        # Core files
        for rel in [
            ".env",
            "api_liveScript.py",
            "equity_tracker.py",
            "sync_fills.py",
            "logger_api.py",
            "run_api_live.sh",
        ]:
            self.check_exists(rel)

        # Joblib and venv
        self.check_exists(self.expected_model)
        self.check_exists("venv", required=False)
        self.check_exists("logs", required=False)

        # Env checks
        self.check_env_value("SYMBOL", self.expected_symbol)
        self.check_env_value("EQUITY_SYMBOL", self.expected_symbol)
        self.check_env_value("FILLS_SYMBOL", self.expected_symbol)
        self.check_env_value("MODEL_PATH", self.expected_model)
        self.check_env_value("FILLS_STATE_PATH", f"fills_state_{self.expected_symbol}.json")
        self.check_env_value("ENABLE_API_LOGGING", allowed=("0", "1"))
        self.check_env_value("ENABLE_CSV_LOGGING", allowed=("0", "1"))
        self.check_env_value("ENABLE_LIVE_TRADING", allowed=("0", "1"))
        self.check_env_contains("BOT_ID", "usdt")
        self.check_path_from_env("MODEL_PATH")
        self.check_env_consistency()

        # Keys/base URL presence only
        for k in ["BINANCE_LIVE_API_KEY", "BINANCE_LIVE_API_SECRET", "API_BASE_URL"]:
            self.check_env_value(k, required=True)

        # Stale old files
        self.check_absent("fills_state_BTCUSDC.json")
        self.check_absent("fills_state_BTCUSDC.backup.json")

        old_trade_log = self.root / "trade_log_live.csv"
        new_trade_log = self.env.get("TRADE_LOG_PATH", "")
        if old_trade_log.exists() and new_trade_log in ("", "trade_log_live.csv"):
            self.add(WARN, "stale:trade_log_live.csv", "Old trade_log_live.csv still present and .env still points to it")
        elif old_trade_log.exists():
            self.add(WARN, "stale:trade_log_live.csv", "Old trade_log_live.csv still present; okay if intentionally archived later")
        else:
            self.add(PASS, "stale:trade_log_live.csv", "Old trade_log_live.csv not present")

        # Python syntax
        for rel in ["api_liveScript.py", "equity_tracker.py", "sync_fills.py", "logger_api.py"]:
            self.check_python_syntax(rel)

        # Content checks
        self.check_text_contains(
            "api_liveScript.py",
            patterns=[
                ('infer_quote_asset', 'quote_inference'),
                ('os.getenv("SYMBOL"', 'env_symbol'),
                ('os.getenv("MODEL_PATH"', 'env_model_path'),
            ],
            must_not_contain=[
                ('SYMBOL = os.getenv("SYMBOL", "BTCUSDC")', 'old_default_symbol'),
                ('MODEL_PATH = os.getenv("MODEL_PATH", "btc_4h_xgb_classifier5k.joblib")', 'old_default_model'),
            ],
        )

        self.check_text_contains(
            "equity_tracker.py",
            patterns=[
                ('get_totals(client, "USDT")', 'usdt_balances'),
                ('equity_usdt', 'equity_usdt_field'),
                ('os.getenv("EQUITY_SYMBOL"', 'equity_env_symbol'),
            ],
            must_not_contain=[
                ('get_totals(client, "USDC")', 'old_usdc_balance'),
                ('equity_usdc', 'old_equity_usdc'),
                ('"BTCUSDC"', 'old_default_btcusdc'),
            ],
        )

        self.check_text_contains(
            "sync_fills.py",
            patterns=[
                ('os.getenv("FILLS_SYMBOL"', 'fills_env_symbol'),
                ('fills_state_{SYMBOL}.json', 'symbol_based_state_path'),
            ],
            must_not_contain=[
                ('SYMBOL = os.getenv("FILLS_SYMBOL", "BTCUSDC")', 'old_default_symbol'),
            ],
        )

        self.check_run_shell_loads_env()

        # Helpful warnings around live trading status
        live_flag = self.env.get("ENABLE_LIVE_TRADING")
        if live_flag == "1":
            self.add(WARN, "env:ENABLE_LIVE_TRADING", "ENABLE_LIVE_TRADING=1. Good for launch, but do manual checks before restoring cron.")
        elif live_flag == "0":
            self.add(PASS, "env:ENABLE_LIVE_TRADING", "ENABLE_LIVE_TRADING=0. Safe for dry-run validation.")

        # Optional: look for obvious old BTCUSDC references in modified runtime files
        scan_files = ["api_liveScript.py", "equity_tracker.py", "sync_fills.py", ".env"]
        for rel in scan_files:
            p = self.root / rel
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            hits = [m.group(0) for m in re.finditer(r"BTCUSDC|USDC", text)]
            if hits:
                self.add(WARN, f"scan:{rel}", f"Still contains {len(hits)} USDC/BTCUSDC reference(s). Review manually.")
            else:
                self.add(PASS, f"scan:{rel}", "No obvious USDC/BTCUSDC references found")

    def print_report(self) -> int:
        widths = {"level": 4, "name": 32}
        print("\n=== USDT preflight report ===")
        print(f"Project root: {self.root}")
        print(f"Expected symbol: {self.expected_symbol}")
        print(f"Expected model : {self.expected_model}\n")

        for r in self.results:
            print(f"[{r.level:<4}] {r.name:<{widths['name']}} {r.detail}")

        fail_count = sum(1 for r in self.results if r.level == FAIL)
        warn_count = sum(1 for r in self.results if r.level == WARN)
        pass_count = sum(1 for r in self.results if r.level == PASS)

        print("\n=== Summary ===")
        print(f"PASS: {pass_count}")
        print(f"WARN: {warn_count}")
        print(f"FAIL: {fail_count}")

        if fail_count:
            print("\nResult: NOT READY")
            return 1
        if warn_count:
            print("\nResult: READY WITH WARNINGS")
            return 0
        print("\nResult: READY")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the project is ready for the BTCUSDT reset/relaunch.")
    parser.add_argument("--root", default=os.getcwd(), help="Project root directory (default: current working directory)")
    parser.add_argument("--model", default="classifier_new_1304.joblib", help="Expected model filename")
    parser.add_argument("--symbol", default="BTCUSDT", help="Expected trading symbol")
    parser.add_argument("--quote", default="USDT", help="Expected quote asset")
    args = parser.parse_args()

    checker = PreflightChecker(Path(args.root).resolve(), args.model, args.symbol, args.quote)
    checker.check_usdt_migration()
    return checker.print_report()


if __name__ == "__main__":
    raise SystemExit(main())

