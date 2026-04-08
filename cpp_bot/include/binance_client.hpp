#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "features.hpp"

namespace bot {

class BinanceClient {
 public:
  BinanceClient(std::string api_key, std::string api_secret);

  std::int64_t get_server_time(bool testnet) const;
  void resync_time();

  std::vector<Candle> fetch_klines_mainnet(const std::string& symbol, const std::string& interval, int limit) const;
  nlohmann::json get_symbol_info(const std::string& symbol) const;
  nlohmann::json get_account() const;
  nlohmann::json get_open_orders(const std::string& symbol) const;
  nlohmann::json get_open_oco_orders() const;
  nlohmann::json get_symbol_ticker_testnet(const std::string& symbol) const;
  nlohmann::json order_market_buy(const std::string& symbol, const std::string& quantity) const;
  nlohmann::json create_oco_order(const std::map<std::string, std::string>& params) const;

 private:
  std::string api_key_;
  std::string api_secret_;
  std::string testnet_base_ = "https://testnet.binance.vision";
  std::string mainnet_base_ = "https://api.binance.com";
  std::int64_t timestamp_offset_ms_ = 0;

  static std::string url_encode(const std::string& value);
  static std::string build_query(const std::map<std::string, std::string>& params);
  std::string sign_query(const std::string& query) const;

  nlohmann::json request_public(const std::string& base, const std::string& path, const std::map<std::string, std::string>& params) const;
  nlohmann::json request_signed(const std::string& method, const std::string& path, std::map<std::string, std::string> params) const;
};

}  // namespace bot
