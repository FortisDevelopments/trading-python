# C++ Translation of `api_testScript.py`

This folder provides a modular C++ implementation of your testnet trading cycle:

- Loads config from `.env`
- Fetches Binance mainnet 4h candles
- Rebuilds indicators/features in C++
- Runs XGBoost inference in C++ using exported tree math
- Places market buy + best-effort OCO on Binance testnet
- Appends run logs to CSV

## 1) Export the Python model for C++ runtime

From repo root:

```bash
python cpp_bot/tools/export_xgb_to_cpp_json.py --joblib btc_4h_xgb_classifier.joblib --out cpp_bot/model_cpp_export.json
```

This converts your `XGBClassifier` trees into a JSON format consumed by C++ (`XgbBinaryModel`).

## 2) Build

```bash
cmake -S cpp_bot -B cpp_bot/build
cmake --build cpp_bot/build --config Release
```

## 3) Configure `.env`

Use existing `.env` keys from your Python bot. C++ runtime reads:

- `BINANCE_TESTNET_API_KEY`
- `BINANCE_TESTNET_API_SECRET`
- `MODEL_JSON_PATH` (default: `model_cpp_export.json`)
- `SYMBOL`, `INTERVAL`, `THRESHOLD`, `HORIZON_STEPS`
- `TAKE_PROFIT`, `STOP_LOSS`, `MAX_OPEN_TRADES`, `PCT_ACCOUNT_PER_TRADE`
- `ENABLE_CSV_LOGGING`, `TRADE_LOG_PATH`

If `MODEL_JSON_PATH` is relative, run from `cpp_bot` or provide absolute path.

## 4) Run

```bash
./cpp_bot/build/trading_bot
```

On Windows (MSVC multi-config):

```bash
./cpp_bot/build/Release/trading_bot.exe
```
