#include "model.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace bot {
namespace {

double sigmoid(double x) {
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

double prob_to_margin(double p) {
  const double eps = 1e-12;
  if (p < eps) {
    p = eps;
  } else if (p > 1.0 - eps) {
    p = 1.0 - eps;
  }
  return std::log(p / (1.0 - p));
}

}  // namespace

XgbBinaryModel XgbBinaryModel::from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open model json: " + path);
  }
  nlohmann::json j;
  in >> j;

  XgbBinaryModel model;
  model.objective_ = j.value("objective", "binary:logistic");
  model.base_score_prob_ = j.value("base_score_prob", 0.5);
  model.feature_names_ = j.at("feature_names").get<std::vector<std::string>>();
  model.trees_.reserve(j.at("trees").size());

  for (const auto& tj : j.at("trees")) {
    Tree t;
    t.left_children = tj.at("left_children").get<std::vector<std::int32_t>>();
    t.right_children = tj.at("right_children").get<std::vector<std::int32_t>>();
    t.split_indices = tj.at("split_indices").get<std::vector<std::int32_t>>();
    t.default_left = tj.at("default_left").get<std::vector<std::uint8_t>>();
    t.is_leaf = tj.at("is_leaf").get<std::vector<std::uint8_t>>();
    t.thresholds = tj.at("thresholds").get<std::vector<double>>();
    t.leaf_values = tj.at("leaf_values").get<std::vector<double>>();
    model.trees_.push_back(std::move(t));
  }
  return model;
}

double XgbBinaryModel::predict_margin(const std::vector<double>& features) const {
  if (features.size() != feature_names_.size()) {
    throw std::runtime_error("Feature size mismatch in model inference.");
  }

  double margin = prob_to_margin(base_score_prob_);
  for (const auto& tree : trees_) {
    std::int32_t node = 0;
    while (true) {
      if (node < 0 || static_cast<std::size_t>(node) >= tree.is_leaf.size()) {
        throw std::runtime_error("Invalid node index while traversing tree.");
      }
      if (tree.is_leaf[node]) {
        margin += tree.leaf_values[node];
        break;
      }
      const auto split_idx = tree.split_indices[node];
      if (split_idx < 0 || static_cast<std::size_t>(split_idx) >= features.size()) {
        throw std::runtime_error("Invalid split index in tree.");
      }
      const double fval = features[split_idx];
      const bool go_left =
          std::isnan(fval) ? static_cast<bool>(tree.default_left[node]) : (fval < tree.thresholds[node]);
      node = go_left ? tree.left_children[node] : tree.right_children[node];
    }
  }
  return margin;
}

double XgbBinaryModel::predict_proba(const std::vector<double>& features) const {
  const double margin = predict_margin(features);
  if (objective_ == "binary:logistic") {
    return sigmoid(margin);
  }
  return margin;
}

}  // namespace bot
