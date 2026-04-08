#pragma once

#include <string>
#include <unordered_map>

namespace bot {

struct Config {
  std::string bot_id = "btc_4h_testnet3";
  bool enable_csv_logging = true;
  std::string log_path = "trade_log.csv";
  std::string model_json_path = "cpp_bot/model_cpp_export.json";
  std::string symbol = "BTCUSDT";
  std::string interval = "4h";
  double threshold = 0.31;
  int horizon_steps = 6;
  double take_profit = 0.070650;
  double stop_loss = 0.007443;
  int max_open_trades = 4;
  double pct_account_per_trade = 0.05;
  std::string testnet_api_key;
  std::string testnet_api_secret;
};

std::unordered_map<std::string, std::string> load_dotenv(const std::string& path);
Config load_config(const std::string& dotenv_path);

}  // namespace bot
