#!/usr/bin/env python3
"""Independent checks for the generated Clean/Hit and ablation statistics."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METHOD_STEMS = {
    "BIMODriver": "bimodriver_original",
    "DISFusion": "disfusion",
    "MNGCL": "mngcl",
}
TASK_STEMS = {"Clean -> Hit": "clean_to_hit", "Hit -> Clean": "hit_to_clean"}


def exact_p(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0, atol=1e-15)]
    ranks = stats.rankdata(np.abs(d), method="average")
    observed = ranks[d > 0].sum()
    null = np.asarray([
        sum(rank for rank, positive in zip(ranks, signs) if positive)
        for signs in product((0, 1), repeat=len(d))
    ])
    center = ranks.sum() / 2.0
    return float(np.mean(np.abs(null - center) >= abs(observed - center) - 1e-12))


def bh(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values, kind="mergesort")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def close(actual: float, expected: float, label: str, atol: float = 1e-12) -> None:
    if not np.isclose(actual, expected, rtol=0, atol=atol):
        raise AssertionError(f"{label}: actual={actual}, expected={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-root", type=Path, default=repo_root / "data" / "leakage_ablation")
    parser.add_argument("--results-dir", type=Path, default=repo_root / "results" / "leakage_ablation")
    args = parser.parse_args()

    benchmark = args.data_root / "2_clean_hit_leakage_benchmark"
    table = pd.read_csv(args.results_dir / "clean_hit_two_sided.csv")
    if len(table) != 8:
        raise AssertionError(f"Expected 8 Clean/Hit rows, got {len(table)}")

    p_values = []
    for index, row in table.iterrows():
        task_stem = TASK_STEMS[row["Task"]]
        baseline = row["Comparison"].split(" - ", 1)[1]
        suffix = row["Metric"].lower()
        left = np.loadtxt(benchmark / f"bimodriver_original_leakage_{task_stem}_{suffix}.txt")
        right = np.loadtxt(benchmark / f"{METHOD_STEMS[baseline]}_leakage_{task_stem}_{suffix}.txt")
        if left.shape != (10,) or right.shape != (10,):
            raise AssertionError("Clean/Hit arrays must each contain 10 run values")
        diff = left - right
        p_value = exact_p(diff)
        p_values.append(p_value)
        close(row["Left mean"], float(left.mean()), f"row {index} left mean")
        close(row["Right mean"], float(right.mean()), f"row {index} right mean")
        close(row["Left SD (ddof=0)"], float(left.std(ddof=0)), f"row {index} left SD")
        close(row["Right SD (ddof=0)"], float(right.std(ddof=0)), f"row {index} right SD")
        close(row["Wilcoxon p (two-sided)"], p_value, f"row {index} p")

    expected_q = bh(np.asarray(p_values))
    for index, (actual, expected) in enumerate(zip(table["BH q"], expected_q)):
        close(float(actual), float(expected), f"row {index} BH q")

    primary = pd.read_csv(args.results_dir / "ablation_10run_primary.csv")
    supplement = pd.read_csv(args.results_dir / "ablation_50fold_supplement.csv")
    if len(primary) != 6 or len(supplement) != 6:
        raise AssertionError("Expected 6 primary and 6 supplementary ablation rows")
    if not (primary["N"] == 10).all() or not (supplement["N"] == 50).all():
        raise AssertionError("Unexpected paired sample size in ablation tables")

    payload = json.loads((args.results_dir / "statistics.json").read_text(encoding="utf-8"))
    if len(payload["tables"]["clean_hit_two_sided"]) != 8:
        raise AssertionError("statistics.json does not contain all Clean/Hit rows")
    if len(payload["input_manifest"]) != 20:
        raise AssertionError("Expected checksums for 20 input metric arrays")
    print("PASS: shapes, fixed-test means/SDs, exact Wilcoxon p-values, BH q-values, and JSON manifest")


if __name__ == "__main__":
    main()
