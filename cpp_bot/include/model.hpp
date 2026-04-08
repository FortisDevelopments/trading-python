#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace bot {

struct Tree {
  std::vector<std::int32_t> left_children;
  std::vector<std::int32_t> right_children;
  std::vector<std::int32_t> split_indices;
  std::vector<std::uint8_t> default_left;
  std::vector<std::uint8_t> is_leaf;
  std::vector<double> thresholds;
  std::vector<double> leaf_values;
};

class XgbBinaryModel {
 public:
  static XgbBinaryModel from_json_file(const std::string& path);
  double predict_margin(const std::vector<double>& features) const;
  double predict_proba(const std::vector<double>& features) const;
  const std::vector<std::string>& feature_names() const { return feature_names_; }

 private:
  double base_score_prob_ = 0.5;
  std::string objective_ = "binary:logistic";
  std::vector<std::string> feature_names_;
  std::vector<Tree> trees_;
};

}  // namespace bot
