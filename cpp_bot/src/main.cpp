#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "binance_client.hpp"
#include "config.hpp"
#include "csv_logger.hpp"
#include "features.hpp"
#include "model.hpp"

namespace bot {
namespace {

double quantize_down(double value, double step) {
  if (step <= 0.0) {
    return value;
  }
  return std::floor(value / step) * step;
}

std::string to_clean_decimal(double value, int precision = 16) {
  std::ostringstream ss;
  ss << std::fixed << std::setprecision(precision) << value;
  std::string s = ss.str();
  while (!s.empty() && s.back() == '0') {
    s.pop_back();
  }
  if (!s.empty() && s.back() == '.') {
    s.pop_back();
  }
  return s.empty() ? "0" : s;
}

std::string now_iso_utc() {
  using namespace std::chrono;
  const auto now = system_clock::now();
  const auto t = system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

std::string ms_to_iso_utc(std::int64_t ms) {
  const std::time_t t = static_cast<std::time_t>(ms / 1000);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream ss;
  ss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
  return ss.str();
}

const nlohmann::json& get_filter(const nlohmann::json& symbol_info, const std::string& filter_type) {
  for (const auto& f : symbol_info.at("symbols").at(0).at("filters")) {
    if (f.at("filterType").get<std::string>() == filter_type) {
      return f;
    }
  }
  throw std::runtime_error("Missing filter type: " + filter_type);
}

double get_free_balance_usdt(const nlohmann::json& account) {
  for (const auto& b : account.at("balances")) {
    if (b.at("asset").get<std::string>() == "USDT") {
      return std::stod(b.at("free").get<std::string>());
    }
  }
  return 0.0;
}

int approx_open_trades(BinanceClient& client, const std::string& symbol) {
  try {
    const auto oco = client.get_open_oco_orders();
    return static_cast<int>(oco.size());
  } catch (...) {
    const auto orders = client.get_open_orders(symbol);
    return static_cast<int>(orders.size() / 2);
  }
}

}  // namespace
}  // namespace bot

int main() {
  using namespace bot;

  try {
    const Config cfg = load_config(".env");
    if (cfg.testnet_api_key.empty() || cfg.testnet_api_secret.empty()) {
      std::cerr << "Missing BINANCE_TESTNET_API_KEY or BINANCE_TESTNET_API_SECRET in .env/environment\n";
      return 2;
    }

    BinanceClient client(cfg.testnet_api_key, cfg.testnet_api_secret);
    client.resync_time();
    std::cout << "Server time synced.\n";

    XgbBinaryModel model = XgbBinaryModel::from_json_file(cfg.model_json_path);
    std::cout << "Model loaded: " << cfg.model_json_path << "\n";

    auto candles = client.fetch_klines_mainnet(cfg.symbol, cfg.interval, 1500);
    auto feats = build_features(candles, cfg.horizon_steps);
    if (feats.empty()) {
      throw std::runtime_error("No feature rows created.");
    }

    const auto& last = feats.back();
    const double p_buy = model.predict_proba(last.values);
    const int signal = (p_buy >= cfg.threshold) ? 1 : 0;
    std::string decision = "none";
    std::string message;

    std::cout << "Features timestamp (UTC): " << ms_to_iso_utc(last.open_time_ms) << "\n";
    std::cout << "Close: " << last.close << "\n";
    std::cout << "P(buy): " << p_buy << "\n";
    std::cout << "Signal: " << signal << "\n";

    if (signal == 1) {
      const auto account = client.get_account();
      const double usdt_free = get_free_balance_usdt(account);
      const int open_trades = approx_open_trades(client, cfg.symbol);
      std::cout << "Open trades (approx): " << open_trades << " / " << cfg.max_open_trades << "\n";

      if (open_trades >= cfg.max_open_trades) {
        decision = "skipped_max_open_trades";
      } else {
        const double usdt_amount = usdt_free * cfg.pct_account_per_trade;
        if (usdt_amount < 10.0) {
          decision = "skipped_small_notional";
        } else {
          const auto symbol_info = client.get_symbol_info(cfg.symbol);
          const auto& lot = get_filter(symbol_info, "LOT_SIZE");
          const auto& price_filter = get_filter(symbol_info, "PRICE_FILTER");
          const double step_size = std::stod(lot.at("stepSize").get<std::string>());
          const double min_qty = std::stod(lot.at("minQty").get<std::string>());
          const double tick_size = std::stod(price_filter.at("tickSize").get<std::string>());

          const auto ticker = client.get_symbol_ticker_testnet(cfg.symbol);
          const double price = std::stod(ticker.at("price").get<std::string>());
          const double raw_qty = usdt_amount / price;
          const double qty = quantize_down(raw_qty, step_size);
          if (qty < min_qty) {
            throw std::runtime_error("Calculated quantity below minQty.");
          }

          const std::string qty_str = to_clean_decimal(qty, 12);
          const auto buy = client.order_market_buy(cfg.symbol, qty_str);
          const double executed_qty = std::stod(buy.value("executedQty", "0"));
          const double quote_spent = std::stod(buy.value("cummulativeQuoteQty", "0"));
          const double entry_price = (executed_qty > 0.0) ? (quote_spent / executed_qty) : std::numeric_limits<double>::quiet_NaN();
          if (!(executed_qty > 0.0) || !std::isfinite(entry_price)) {
            throw std::runtime_error("Buy order did not execute correctly.");
          }

          std::cout << "Bought qty=" << executed_qty << " @ avg entry=" << entry_price << "\n";

          const double tp_raw = entry_price * (1.0 + cfg.take_profit);
          const double sl_stop_raw = entry_price * (1.0 - cfg.stop_loss);
          const double sl_limit_raw = sl_stop_raw * (1.0 - 0.001);
          const double tp = quantize_down(tp_raw, tick_size);
          const double sl_stop = quantize_down(sl_stop_raw, tick_size);
          const double sl_limit = quantize_down(sl_limit_raw, tick_size);

          try {
            const auto oco = client.create_oco_order({
                {"symbol", cfg.symbol},
                {"side", "SELL"},
                {"quantity", qty_str},
                {"aboveType", "LIMIT_MAKER"},
                {"abovePrice", to_clean_decimal(tp, 8)},
                {"belowType", "STOP_LOSS_LIMIT"},
                {"belowStopPrice", to_clean_decimal(sl_stop, 8)},
                {"belowPrice", to_clean_decimal(sl_limit, 8)},
                {"belowTimeInForce", "GTC"},
            });
            (void)oco;
            decision = "bought";
          } catch (const std::exception& e) {
            decision = "bought_oco_failed";
            message = e.what();
          }
        }
      }
    }

    if (cfg.enable_csv_logging) {
      append_csv_rows(
          cfg.log_path,
          {"event", "ts", "candle_ts", "close", "p_buy", "signal", "decision", "message"},
          {{
              "run",
              now_iso_utc(),
              ms_to_iso_utc(last.open_time_ms),
              to_clean_decimal(last.close, 8),
              to_clean_decimal(p_buy, 12),
              std::to_string(signal),
              decision,
              message,
          }});
    }

    std::cout << "DONE. action: " << decision << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
