#include "csv_logger.hpp"

#include <filesystem>
#include <fstream>

namespace bot {
namespace {

std::string escape_csv(const std::string& s) {
  bool must_quote = false;
  for (char c : s) {
    if (c == '"' || c == ',' || c == '\n' || c == '\r') {
      must_quote = true;
      break;
    }
  }
  if (!must_quote) {
    return s;
  }
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') {
      out += "\"\"";
    } else {
      out += c;
    }
  }
  out += "\"";
  return out;
}

}  // namespace

void append_csv_rows(const std::string& path, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows) {
  if (rows.empty()) {
    return;
  }
  const bool exists = std::filesystem::exists(path);
  std::ofstream out(path, std::ios::app);
  if (!out.is_open()) {
    return;
  }
  if (!exists && !headers.empty()) {
    for (std::size_t i = 0; i < headers.size(); ++i) {
      if (i) {
        out << ",";
      }
      out << escape_csv(headers[i]);
    }
    out << "\n";
  }
  for (const auto& row : rows) {
    for (std::size_t i = 0; i < row.size(); ++i) {
      if (i) {
        out << ",";
      }
      out << escape_csv(row[i]);
    }
    out << "\n";
  }
}

}  // namespace bot
