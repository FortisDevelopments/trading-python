#include "features.hpp"

#include <cmath>
#include <ctime>
#include <limits>

namespace bot {

const std::vector<std::string> kFeatureNames = {
    "Volume",
    "returns",
    "log_returns",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "PROC_HORIZON",
    "hour",
    "ADX_14",
};

namespace {

double nanv() {
  return std::numeric_limits<double>::quiet_NaN();
}

bool is_valid(double v) {
  return std::isfinite(v);
}

std::vector<double> rolling_mean(const std::vector<double>& x, int w) {
  std::vector<double> out(x.size(), nanv());
  if (w <= 0) {
    return out;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (is_valid(x[i])) {
      sum += x[i];
    }
    if (i >= static_cast<std::size_t>(w) && is_valid(x[i - w])) {
      sum -= x[i - w];
    }
    if (i + 1 >= static_cast<std::size_t>(w)) {
      out[i] = sum / static_cast<double>(w);
    }
  }
  return out;
}

std::vector<double> rolling_sum(const std::vector<double>& x, int w) {
  std::vector<double> out(x.size(), nanv());
  if (w <= 0) {
    return out;
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (is_valid(x[i])) {
      sum += x[i];
    }
    if (i >= static_cast<std::size_t>(w) && is_valid(x[i - w])) {
      sum -= x[i - w];
    }
    if (i + 1 >= static_cast<std::size_t>(w)) {
      out[i] = sum;
    }
  }
  return out;
}

std::vector<double> ema(const std::vector<double>& x, int span) {
  std::vector<double> out(x.size(), nanv());
  if (x.empty()) {
    return out;
  }
  const double alpha = 2.0 / (static_cast<double>(span) + 1.0);
  out[0] = x[0];
  for (std::size_t i = 1; i < x.size(); ++i) {
    out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1];
  }
  return out;
}

int hour_utc(std::int64_t ms) {
  const std::time_t secs = static_cast<std::time_t>(ms / 1000);
  std::tm tmv{};
#ifdef _WIN32
  gmtime_s(&tmv, &secs);
#else
  gmtime_r(&secs, &tmv);
#endif
  return tmv.tm_hour;
}

}  // namespace

std::vector<FeatureRow> build_features(const std::vector<Candle>& candles, int horizon_steps) {
  const std::size_t n = candles.size();
  if (n < 100) {
    return {};
  }

  std::vector<double> close(n), high(n), low(n), volume(n);
  std::vector<double> hour(n), returns(n, nanv()), log_returns(n, nanv()), proc_horizon(n, nanv());
  for (std::size_t i = 0; i < n; ++i) {
    close[i] = candles[i].close;
    high[i] = candles[i].high;
    low[i] = candles[i].low;
    volume[i] = candles[i].volume;
    hour[i] = static_cast<double>(hour_utc(candles[i].open_time_ms));
    if (i > 0 && close[i - 1] != 0.0) {
      returns[i] = close[i] / close[i - 1] - 1.0;
      log_returns[i] = std::log(close[i] / close[i - 1]);
    }
    if (i >= static_cast<std::size_t>(horizon_steps) && close[i - horizon_steps] != 0.0) {
      proc_horizon[i] = close[i] / close[i - horizon_steps] - 1.0;
    }
  }

  // RSI(14)
  std::vector<double> delta(n, nanv()), gain(n, 0.0), loss(n, 0.0), rsi(n, nanv());
  for (std::size_t i = 1; i < n; ++i) {
    delta[i] = close[i] - close[i - 1];
    gain[i] = delta[i] > 0.0 ? delta[i] : 0.0;
    loss[i] = delta[i] < 0.0 ? -delta[i] : 0.0;
  }
  auto avg_gain = rolling_mean(gain, 14);
  auto avg_loss = rolling_mean(loss, 14);
  for (std::size_t i = 0; i < n; ++i) {
    if (is_valid(avg_gain[i]) && is_valid(avg_loss[i]) && avg_loss[i] != 0.0) {
      const double rs = avg_gain[i] / avg_loss[i];
      rsi[i] = 100.0 - (100.0 / (1.0 + rs));
    }
  }

  // MACD(12,26,9)
  auto ema12 = ema(close, 12);
  auto ema26 = ema(close, 26);
  std::vector<double> macd(n, nanv());
  for (std::size_t i = 0; i < n; ++i) {
    macd[i] = ema12[i] - ema26[i];
  }
  auto macd_signal = ema(macd, 9);

  // ADX(14) rolling approximation to match Python code.
  std::vector<double> tr(n, nanv()), up_move(n, nanv()), down_move(n, nanv());
  std::vector<double> plus_dm(n, 0.0), minus_dm(n, 0.0), adx(n, nanv());
  for (std::size_t i = 1; i < n; ++i) {
    const double tr1 = high[i] - low[i];
    const double tr2 = std::fabs(high[i] - close[i - 1]);
    const double tr3 = std::fabs(low[i] - close[i - 1]);
    tr[i] = std::fmax(tr1, std::fmax(tr2, tr3));
    up_move[i] = high[i] - high[i - 1];
    down_move[i] = low[i - 1] - low[i];
    plus_dm[i] = (up_move[i] > down_move[i] && up_move[i] > 0.0) ? up_move[i] : 0.0;
    minus_dm[i] = (down_move[i] > up_move[i] && down_move[i] > 0.0) ? down_move[i] : 0.0;
  }

  auto atr = rolling_mean(tr, 14);
  auto plus_dm_14 = rolling_sum(plus_dm, 14);
  auto minus_dm_14 = rolling_sum(minus_dm, 14);
  std::vector<double> dx(n, nanv());
  for (std::size_t i = 0; i < n; ++i) {
    if (is_valid(atr[i]) && atr[i] != 0.0) {
      const double plus_di = 100.0 * (plus_dm_14[i] / atr[i]);
      const double minus_di = 100.0 * (minus_dm_14[i] / atr[i]);
      const double denom = plus_di + minus_di;
      if (denom != 0.0) {
        dx[i] = std::fabs(plus_di - minus_di) / denom * 100.0;
      }
    }
  }
  adx = rolling_mean(dx, 14);

  std::vector<FeatureRow> rows;
  rows.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> vals = {
        volume[i], returns[i], log_returns[i], rsi[i], macd[i], macd_signal[i], proc_horizon[i], hour[i], adx[i]};
    bool ok = true;
    for (double v : vals) {
      if (!is_valid(v)) {
        ok = false;
        break;
      }
    }
    if (!ok) {
      continue;
    }
    rows.push_back(FeatureRow{candles[i].open_time_ms, close[i], std::move(vals)});
  }

  return rows;
}

}  // namespace bot
