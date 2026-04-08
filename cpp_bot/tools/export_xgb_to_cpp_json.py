#!/usr/bin/env python3
"""
Export a joblib XGBClassifier into a compact JSON format for native C++ inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib


def _sanitize_base_score(value: str | float | int) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return float(s)


def _fill_node(arrays: dict, node: dict, feature_to_idx: dict[str, int]) -> None:
    node_id = int(node["nodeid"])
    arrays["is_leaf"][node_id] = 1 if "leaf" in node else 0
    if "leaf" in node:
        arrays["leaf_values"][node_id] = float(node["leaf"])
        return

    split = node["split"]
    if isinstance(split, str):
        if split.startswith("f") and split[1:].isdigit():
            split_idx = int(split[1:])
        elif split in feature_to_idx:
            split_idx = feature_to_idx[split]
        else:
            split_idx = int(split)
    else:
        split_idx = int(split)

    yes = int(node["yes"])
    no = int(node["no"])
    missing = int(node["missing"])

    arrays["split_indices"][node_id] = split_idx
    arrays["thresholds"][node_id] = float(node["split_condition"])
    arrays["left_children"][node_id] = yes
    arrays["right_children"][node_id] = no
    arrays["default_left"][node_id] = 1 if missing == yes else 0

    for child in node.get("children", []):
        _fill_node(arrays, child, feature_to_idx)


def _convert_tree(tree_obj: dict, feature_to_idx: dict[str, int]) -> dict:
    max_id = 0
    stack = [tree_obj]
    while stack:
        nd = stack.pop()
        max_id = max(max_id, int(nd["nodeid"]))
        stack.extend(nd.get("children", []))

    size = max_id + 1
    arrays = {
        "left_children": [-1] * size,
        "right_children": [-1] * size,
        "split_indices": [-1] * size,
        "default_left": [0] * size,
        "is_leaf": [0] * size,
        "thresholds": [0.0] * size,
        "leaf_values": [0.0] * size,
    }
    _fill_node(arrays, tree_obj, feature_to_idx)
    return arrays


def export_model(joblib_path: Path, output_path: Path) -> None:
    model = joblib.load(joblib_path)
    booster = model.get_booster()
    config = json.loads(booster.save_config())
    dump = booster.get_dump(dump_format="json")

    objective = config["learner"]["objective"]["name"]
    base_score_raw = config["learner"]["learner_model_param"]["base_score"]
    base_score_prob = _sanitize_base_score(base_score_raw)

    if not math.isfinite(base_score_prob) or base_score_prob <= 0 or base_score_prob >= 1:
        base_score_prob = 0.5

    feature_names = [str(x) for x in list(getattr(model, "feature_names_in_", []))]

    if not feature_names:
        # Fallback to f0..fn if feature names are unavailable.
        n_features = int(getattr(model, "n_features_in_", 0))
        feature_names = [f"f{i}" for i in range(n_features)]

    feature_to_idx = {name: i for i, name in enumerate(feature_names)}
    trees = [_convert_tree(json.loads(tree), feature_to_idx) for tree in dump]

    out = {
        "objective": objective,
        "base_score_prob": base_score_prob,
        "feature_names": feature_names,
        "trees": trees,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"Exported {len(trees)} trees to: {output_path}")
    print(f"Objective: {objective}")
    print(f"Features: {feature_names}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joblib", required=True, help="Path to .joblib XGBClassifier")
    parser.add_argument("--out", required=True, help="Output JSON path for C++ runtime")
    args = parser.parse_args()

    export_model(Path(args.joblib), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
