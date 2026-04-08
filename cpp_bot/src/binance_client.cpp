#include "binance_client.hpp"

#include <cctype>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <cpr/cpr.h>
#include <openssl/hmac.h>

namespace bot {
namespace {

std::int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

void ensure_ok(const cpr::Response& r, const std::string& context) {
  if (r.error.code != cpr::ErrorCode::OK) {
    throw std::runtime_error(context + " request error: " + r.error.message);
  }
  if (r.status_code < 200 || r.status_code >= 300) {
    throw std::runtime_error(context + " status=" + std::to_string(r.status_code) + " body=" + r.text);
  }
}

}  // namespace

BinanceClient::BinanceClient(std::string api_key, std::string api_secret)
    : api_key_(std::move(api_key)), api_secret_(std::move(api_secret)) {}

std::string BinanceClient::url_encode(const std::string& value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << std::hex << std::uppercase;
  for (const unsigned char c : value) {
    if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      escaped << c;
    } else {
      escaped << '%' << std::setw(2) << static_cast<int>(c);
    }
  }
  return escaped.str();
}

std::string BinanceClient::build_query(const std::map<std::string, std::string>& params) {
  std::ostringstream ss;
  bool first = true;
  for (const auto& kv : params) {
    if (!first) {
      ss << '&';
    }
    first = false;
    ss << url_encode(kv.first) << '=' << url_encode(kv.second);
  }
  return ss.str();
}

std::string BinanceClient::sign_query(const std::string& query) const {
  unsigned char out[EVP_MAX_MD_SIZE];
  unsigned int out_len = 0;
  HMAC(
      EVP_sha256(),
      api_secret_.data(),
      static_cast<int>(api_secret_.size()),
      reinterpret_cast<const unsigned char*>(query.data()),
      query.size(),
      out,
      &out_len);
  std::ostringstream ss;
  ss << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < out_len; ++i) {
    ss << std::setw(2) << static_cast<int>(out[i]);
  }
  return ss.str();
}

nlohmann::json BinanceClient::request_public(
    const std::string& base, const std::string& path, const std::map<std::string, std::string>& params) const {
  const auto query = build_query(params);
  const std::string url = base + path + (query.empty() ? "" : "?" + query);
  const auto r = cpr::Get(cpr::Url{url});
  ensure_ok(r, "public");
  return nlohmann::json::parse(r.text);
}

nlohmann::json BinanceClient::request_signed(
    const std::string& method, const std::string& path, std::map<std::string, std::string> params) const {
  params["timestamp"] = std::to_string(now_ms() + timestamp_offset_ms_);
  params["recvWindow"] = "5000";
  const auto query = build_query(params);
  const auto signature = sign_query(query);
  const auto final_query = query + "&signature=" + signature;
  const auto url = testnet_base_ + path + "?" + final_query;
  const cpr::Header headers = {{"X-MBX-APIKEY", api_key_}};

  cpr::Response r;
  if (method == "GET") {
    r = cpr::Get(cpr::Url{url}, headers);
  } else if (method == "POST") {
    r = cpr::Post(cpr::Url{url}, headers);
  } else {
    throw std::runtime_error("Unsupported HTTP method: " + method);
  }
  ensure_ok(r, "signed");
  return nlohmann::json::parse(r.text);
}

std::int64_t BinanceClient::get_server_time(bool testnet) const {
  const auto j = request_public(testnet ? testnet_base_ : mainnet_base_, "/api/v3/time", {});
  return j.at("serverTime").get<std::int64_t>();
}

void BinanceClient::resync_time() {
  timestamp_offset_ms_ = get_server_time(true) - now_ms();
}

std::vector<Candle> BinanceClient::fetch_klines_mainnet(const std::string& symbol, const std::string& interval, int limit) const {
  const auto j = request_public(mainnet_base_, "/api/v3/klines", {
                                                              {"symbol", symbol},
                                                              {"interval", interval},
                                                              {"limit", std::to_string(limit)},
                                                          });
  std::vector<Candle> candles;
  candles.reserve(j.size());
  const auto now = now_ms();
  for (const auto& row : j) {
    Candle c;
    c.open_time_ms = row.at(0).get<std::int64_t>();
    c.open = std::stod(row.at(1).get<std::string>());
    c.high = std::stod(row.at(2).get<std::string>());
    c.low = std::stod(row.at(3).get<std::string>());
    c.close = std::stod(row.at(4).get<std::string>());
    c.volume = std::stod(row.at(5).get<std::string>());
    c.close_time_ms = row.at(6).get<std::int64_t>();
    if (c.close_time_ms <= now) {
      candles.push_back(c);
    }
  }
  return candles;
}

nlohmann::json BinanceClient::get_symbol_info(const std::string& symbol) const {
  return request_public(testnet_base_, "/api/v3/exchangeInfo", {{"symbol", symbol}});
}

nlohmann::json BinanceClient::get_account() const {
  return request_signed("GET", "/api/v3/account", {});
}

nlohmann::json BinanceClient::get_open_orders(const std::string& symbol) const {
  return request_signed("GET", "/api/v3/openOrders", {{"symbol", symbol}});
}

nlohmann::json BinanceClient::get_open_oco_orders() const {
  return request_signed("GET", "/api/v3/openOrderList", {});
}

nlohmann::json BinanceClient::get_symbol_ticker_testnet(const std::string& symbol) const {
  return request_public(testnet_base_, "/api/v3/ticker/price", {{"symbol", symbol}});
}

nlohmann::json BinanceClient::order_market_buy(const std::string& symbol, const std::string& quantity) const {
  return request_signed("POST", "/api/v3/order", {
                                                   {"symbol", symbol},
                                                   {"side", "BUY"},
                                                   {"type", "MARKET"},
                                                   {"quantity", quantity},
                                               });
}

nlohmann::json BinanceClient::create_oco_order(const std::map<std::string, std::string>& params) const {
  return request_signed("POST", "/api/v3/orderList/oco", params);
}

}  // namespace bot
