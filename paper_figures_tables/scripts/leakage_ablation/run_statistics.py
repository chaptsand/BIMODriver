#!/usr/bin/env python3
"""Recompute the Clean/Hit and four-way ablation statistics.

The Clean/Hit analysis uses the final fixed-test arrays, paired by run (n=10).
The primary ablation analysis uses each run's five-fold mean (n=10). A direct
50-fold analysis is emitted only as a supplementary sensitivity analysis.
All tests in this workflow are two-sided paired Wilcoxon signed-rank tests.
"""

from __future__ import annotations

import argparse
import hashlib
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
ABLATION_STEMS = {
    "Original Transductive": "pan-cancer",
    "Masked Transductive": "pan-cancer_masked",
    "Original Strict Inductive": "pan-cancer_inductive",
    "Masked Strict Inductive": "pan-cancer_masked_inductive",
}


def load_array(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(np.loadtxt(path, dtype=float), dtype=float)
    if values.shape != expected_shape:
        raise ValueError(f"{path}: expected {expected_shape}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: non-finite value detected")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_signed_rank_p(diff: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value, including tied absolute ranks."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0, atol=1e-15)]
    if not len(d):
        return 1.0
    if len(d) > 20:
        raise ValueError("Exact enumeration is limited to at most 20 nonzero pairs")
    ranks = stats.rankdata(np.abs(d), method="average")
    observed = float(ranks[d > 0].sum())
    null = np.asarray(
        [sum(rank for rank, positive in zip(ranks, signs) if positive)
         for signs in product((0, 1), repeat=len(d))],
        dtype=float,
    )
    center = float(ranks.sum()) / 2.0
    return float(np.mean(np.abs(null - center) >= abs(observed - center) - 1e-12))


def asymptotic_signed_rank_p(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    if np.all(np.isclose(d, 0.0, atol=1e-15)):
        return 1.0
    return float(stats.wilcoxon(
        d,
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
        method="asymptotic",
    ).pvalue)


def rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0, atol=1e-15)]
    if not len(d):
        return 0.0
    ranks = stats.rankdata(np.abs(d), method="average")
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    return (positive - negative) / (positive + negative)


def hodges_lehmann(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def bootstrap_hl_ci(
    diff: np.ndarray,
    n_bootstrap: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the paired Hodges-Lehmann difference."""
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(len(d))
    values = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, chunk_size):
        stop = min(start + chunk_size, n_bootstrap)
        sample = d[rng.integers(0, len(d), size=(stop - start, len(d)))]
        values[start:stop] = np.median((sample[:, i] + sample[:, j]) / 2.0, axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def bh_adjust(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def deterministic_seed(key: str, base_seed: int) -> int:
    token = hashlib.sha256(f"{base_seed}|{key}".encode()).hexdigest()[:8]
    return int(token, 16)


def comparison_row(
    task: str,
    metric: str,
    left_name: str,
    right_name: str,
    left: np.ndarray,
    right: np.ndarray,
    paired_unit: str,
    exact: bool,
    n_bootstrap: int,
    base_seed: int,
) -> dict[str, object]:
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Unpaired arrays: {left.shape} and {right.shape}")
    diff = left - right
    key = f"{task}|{metric}|{left_name}|{right_name}|{paired_unit}"
    ci_low, ci_high = bootstrap_hl_ci(
        diff,
        n_bootstrap=n_bootstrap,
        seed=deterministic_seed(key, base_seed),
    )
    return {
        "Task": task,
        "Metric": metric,
        "Comparison": f"{left_name} - {right_name}",
        "Paired unit": paired_unit,
        "N": int(len(diff)),
        "Left mean": float(left.mean()),
        "Left SD (ddof=0)": float(left.std(ddof=0)),
        "Right mean": float(right.mean()),
        "Right SD (ddof=0)": float(right.std(ddof=0)),
        "Mean difference": float(diff.mean()),
        "Wilcoxon p (two-sided)": (
            exact_signed_rank_p(diff) if exact else asymptotic_signed_rank_p(diff)
        ),
        "Effect size r_rb": rank_biserial(diff),
        "HL difference": hodges_lehmann(diff),
        "95% CI low": ci_low,
        "95% CI high": ci_high,
    }


def add_family_bh(rows: list[dict[str, object]]) -> None:
    q_values = bh_adjust([float(row["Wilcoxon p (two-sided)"]) for row in rows])
    for row, q_value in zip(rows, q_values):
        row["BH q"] = float(q_value)
        row["Significant (q<0.05)"] = bool(q_value < 0.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-root",
        type=Path,
        default=repo_root / "data" / "leakage_ablation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "leakage_ablation",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pan_dir = args.data_root / "1_pan_cancer_10x5_cv"
    benchmark_dir = args.data_root / "2_clean_hit_leakage_benchmark"
    input_files: set[Path] = set()

    # Fixed-test Clean/Hit benchmark: 8 comparisons in one BH family.
    fixed_test: dict[tuple[str, str, str], np.ndarray] = {}
    raw_clean_hit: list[dict[str, object]] = []
    for method, method_stem in METHOD_STEMS.items():
        for task, task_stem in TASK_STEMS.items():
            for metric, suffix in (("AUROC", "auroc"), ("AUPRC", "auprc")):
                path = benchmark_dir / f"{method_stem}_leakage_{task_stem}_{suffix}.txt"
                values = load_array(path, (10,))
                input_files.add(path)
                fixed_test[(method, task, metric)] = values
                raw_clean_hit.extend({
                    "Method": method,
                    "Task": task,
                    "Metric": metric,
                    "Run": run + 1,
                    "Fixed-test value": float(value),
                } for run, value in enumerate(values))

    clean_hit_rows: list[dict[str, object]] = []
    for task in TASK_STEMS:
        for metric in ("AUROC", "AUPRC"):
            left = fixed_test[("BIMODriver", task, metric)]
            for baseline in ("DISFusion", "MNGCL"):
                clean_hit_rows.append(comparison_row(
                    task,
                    metric,
                    "BIMODriver",
                    baseline,
                    left,
                    fixed_test[(baseline, task, metric)],
                    "run on the same fixed test set",
                    exact=True,
                    n_bootstrap=args.bootstrap,
                    base_seed=args.seed,
                ))
    add_family_bh(clean_hit_rows)

    # Four-way 10x5 ablation: run means are primary; 50 folds are supplemental.
    matrices: dict[tuple[str, str], np.ndarray] = {}
    raw_ablation: list[dict[str, object]] = []
    for configuration, stem in ABLATION_STEMS.items():
        for metric, suffix in (("AUROC", "auroc"), ("AUPRC", "auprc")):
            path = pan_dir / f"{stem}_{suffix}.txt"
            matrix = load_array(path, (10, 5))
            input_files.add(path)
            matrices[(configuration, metric)] = matrix
            for run in range(10):
                for fold in range(5):
                    raw_ablation.append({
                        "Configuration": configuration,
                        "Metric": metric,
                        "Run": run + 1,
                        "Fold": fold + 1,
                        "Fold value": float(matrix[run, fold]),
                        "Run mean": float(matrix[run].mean()),
                    })

    ablation_run_rows: list[dict[str, object]] = []
    ablation_fold_rows: list[dict[str, object]] = []
    for metric in ("AUROC", "AUPRC"):
        baseline = matrices[("Original Transductive", metric)]
        for candidate in (
            "Masked Transductive",
            "Original Strict Inductive",
            "Masked Strict Inductive",
        ):
            comparison = matrices[(candidate, metric)]
            ablation_run_rows.append(comparison_row(
                "Four-way ablation",
                metric,
                candidate,
                "Original Transductive",
                comparison.mean(axis=1),
                baseline.mean(axis=1),
                "run mean (mean of 5 folds)",
                exact=True,
                n_bootstrap=args.bootstrap,
                base_seed=args.seed,
            ))
            ablation_fold_rows.append(comparison_row(
                "Four-way ablation (supplement)",
                metric,
                candidate,
                "Original Transductive",
                comparison.reshape(-1),
                baseline.reshape(-1),
                "coordinate-matched fold value",
                exact=False,
                n_bootstrap=args.bootstrap,
                base_seed=args.seed,
            ))
    add_family_bh(ablation_run_rows)
    add_family_bh(ablation_fold_rows)

    tables = {
        "clean_hit_two_sided": pd.DataFrame(clean_hit_rows),
        "ablation_10run_primary": pd.DataFrame(ablation_run_rows),
        "ablation_50fold_supplement": pd.DataFrame(ablation_fold_rows),
        "raw_clean_hit": pd.DataFrame(raw_clean_hit),
        "raw_ablation": pd.DataFrame(raw_ablation),
    }
    for name, frame in tables.items():
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)

    manifest = {
        str(path.relative_to(args.data_root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(input_files)
    }
    payload = {
        "method": {
            "clean_hit": "two-sided paired Wilcoxon on 10 fixed-test run values; BH over 8 comparisons",
            "ablation_primary": "two-sided paired Wilcoxon on 10 run means; BH over 6 comparisons",
            "ablation_supplement": "two-sided asymptotic paired Wilcoxon on 50 fold values; BH over 6 comparisons",
            "effect_size": "paired rank-biserial correlation",
            "estimate": "paired Hodges-Lehmann difference (left minus right)",
            "confidence_interval": f"percentile bootstrap, {args.bootstrap} resamples",
            "descriptive_sd": "population SD (ddof=0), matching the supplied experiment summaries",
        },
        "tables": {name: frame.to_dict(orient="records") for name, frame in tables.items()},
        "input_manifest": manifest,
        "seed": args.seed,
    }
    (args.output_dir / "statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote reproducible outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
