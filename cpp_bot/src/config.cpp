#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace bot {
namespace {

std::string trim(const std::string& s) {
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::string get_value(
    const std::unordered_map<std::string, std::string>& dotenv,
    const std::string& key,
    const std::string& fallback) {
  if (const char* env = std::getenv(key.c_str())) {
    return std::string(env);
  }
  const auto it = dotenv.find(key);
  if (it != dotenv.end()) {
    return it->second;
  }
  return fallback;
}

bool as_bool(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

}  // namespace

std::unordered_map<std::string, std::string> load_dotenv(const std::string& path) {
  std::unordered_map<std::string, std::string> env;
  std::ifstream in(path);
  if (!in.is_open()) {
    return env;
  }
  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
      continue;
    }
    auto key = trim(line.substr(0, pos));
    auto val = trim(line.substr(pos + 1));
    if (!val.empty() && ((val.front() == '"' && val.back() == '"') || (val.front() == '\'' && val.back() == '\''))) {
      val = val.substr(1, val.size() - 2);
    }
    env[key] = val;
  }
  return env;
}

Config load_config(const std::string& dotenv_path) {
  const auto dotenv = load_dotenv(dotenv_path);
  Config cfg;
  cfg.bot_id = get_value(dotenv, "BOT_ID", cfg.bot_id);
  cfg.enable_csv_logging = as_bool(get_value(dotenv, "ENABLE_CSV_LOGGING", "1"));
  cfg.log_path = get_value(dotenv, "TRADE_LOG_PATH", cfg.log_path);
  cfg.model_json_path = get_value(dotenv, "MODEL_JSON_PATH", cfg.model_json_path);
  cfg.symbol = get_value(dotenv, "SYMBOL", cfg.symbol);
  cfg.interval = get_value(dotenv, "INTERVAL", cfg.interval);
  cfg.threshold = std::stod(get_value(dotenv, "THRESHOLD", std::to_string(cfg.threshold)));
  cfg.horizon_steps = std::stoi(get_value(dotenv, "HORIZON_STEPS", std::to_string(cfg.horizon_steps)));
  cfg.take_profit = std::stod(get_value(dotenv, "TAKE_PROFIT", std::to_string(cfg.take_profit)));
  cfg.stop_loss = std::stod(get_value(dotenv, "STOP_LOSS", std::to_string(cfg.stop_loss)));
  cfg.max_open_trades = std::stoi(get_value(dotenv, "MAX_OPEN_TRADES", std::to_string(cfg.max_open_trades)));
  cfg.pct_account_per_trade = std::stod(get_value(dotenv, "PCT_ACCOUNT_PER_TRADE", std::to_string(cfg.pct_account_per_trade)));
  cfg.testnet_api_key = get_value(dotenv, "BINANCE_TESTNET_API_KEY", "");
  cfg.testnet_api_secret = get_value(dotenv, "BINANCE_TESTNET_API_SECRET", "");
  return cfg;
}

}  // namespace bot
