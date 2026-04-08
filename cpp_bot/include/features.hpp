#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bot {

struct Candle {
  std::int64_t open_time_ms{};
  std::int64_t close_time_ms{};
  double open{};
  double high{};
  double low{};
  double close{};
  double volume{};
};

struct FeatureRow {
  std::int64_t open_time_ms{};
  double close{};
  std::vector<double> values;
};

constexpr int kFeatureCount = 9;
extern const std::vector<std::string> kFeatureNames;

std::vector<FeatureRow> build_features(const std::vector<Candle>& candles, int horizon_steps);

}  // namespace bot
