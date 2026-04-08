#pragma once

#include <string>
#include <vector>

namespace bot {

void append_csv_rows(const std::string& path, const std::vector<std::string>& headers, const std::vector<std::vector<std::string>>& rows);

}  // namespace bot
