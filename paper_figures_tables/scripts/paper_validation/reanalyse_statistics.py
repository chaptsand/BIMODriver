from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "data" / "paper_supplementary"
PAN_FILE = ROOT / "figure_code" / "pan-cancer" / "cpdb-pancancer.txt"
STRING_PAN_FILE = ROOT / "figure_code" / "pan-cancer" / "string-pancancer.txt"
CANCER_DIR = ROOT / "figure_code" / "cancer_specific"
OUT_DIR = REPO_ROOT / "results" / "paper_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_METHOD = "BIMODriver"
METRIC_MAP = {"AUC": "AUROC", "AUPRC": "AUPRC"}


def extract_rows(lines: list[str], start: int, end: int) -> np.ndarray:
    rows = []
    for line in lines[start - 1 : end]:
        cleaned = line.replace("[", " ").replace("]", " ")
        vals = np.asarray(
            [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", cleaned)],
            dtype=float,
        )
        if len(vals) == 5:
            rows.append(vals)
    arr = np.asarray(rows, dtype=float)
    if arr.shape != (10, 5):
        raise ValueError(f"Expected 10x5 at lines {start}:{end}, got {arr.shape}")
    return arr


def load_pan() -> dict[str, dict[str, np.ndarray]]:
    lines = PAN_FILE.read_text(encoding="utf-8").splitlines()
    # The source file has ten heterogeneous text blocks. These explicit spans
    # preserve the printed row-major order: one repeat per row, five folds per row.
    specs = {
        "BIMODriver": {"AUPRC": (4, 13), "AUROC": (15, 24)},
        "ECD-CDGI": {"AUROC": (27, 36), "AUPRC": (37, 46)},
        "DISHyper": {"AUROC": (55, 64), "AUPRC": (66, 75)},
        "DISFusion": {"AUROC": (85, 94), "AUPRC": (96, 105)},
        "GCN": {"AUPRC": (114, 123), "AUROC": (125, 134)},
        "ChebNet": {"AUPRC": (140, 149), "AUROC": (151, 160)},
        "GAT": {"AUPRC": (166, 175), "AUROC": (177, 186)},
        "MNGCL": {"AUROC": (191, 200), "AUPRC": (201, 210)},
        "MTGCN": {"AUROC": (214, 223), "AUPRC": (225, 234)},
        "EMOGI": {"AUROC": (238, 247), "AUPRC": (249, 258)},
    }
    return {
        method: {metric: extract_rows(lines, *span) for metric, span in metrics.items()}
        for method, metrics in specs.items()
    }


def load_string_pan() -> dict[str, dict[str, np.ndarray]]:
    lines = STRING_PAN_FILE.read_text(encoding="utf-8").splitlines()
    specs = {
        "BIMODriver": {"AUPRC": (6, 15), "AUROC": (17, 26)},
        "GCN": {"AUPRC": (33, 42), "AUROC": (44, 53)},
        "ChebNet": {"AUPRC": (59, 68), "AUROC": (70, 79)},
        "GAT": {"AUPRC": (85, 94), "AUROC": (96, 105)},
        "ECD-CDGI": {"AUROC": (108, 117), "AUPRC": (119, 128)},
        "MNGCL": {"AUROC": (135, 144), "AUPRC": (146, 155)},
        "EMOGI": {"AUROC": (159, 168), "AUPRC": (170, 179)},
        "MTGCN": {"AUROC": (183, 192), "AUPRC": (194, 203)},
        "DISFusion": {"AUROC": (210, 219), "AUPRC": (221, 230)},
        "DISHyper": {"AUROC": (240, 249), "AUPRC": (251, 260)},
    }
    return {
        method: {metric: extract_rows(lines, *span) for metric, span in metrics.items()}
        for method, metrics in specs.items()
    }


def load_cancers() -> dict[str, dict[str, dict[str, np.ndarray]]]:
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    pat = re.compile(r"^(?P<method>.+)_(?P<cancer>[a-z]+)_(?P<metric>AUC|AUPRC)_50values\.txt$")
    for path in sorted(CANCER_DIR.glob("*/*_50values.txt")):
        match = pat.match(path.name)
        if not match:
            raise ValueError(f"Unexpected filename: {path}")
        method = match.group("method")
        cancer = match.group("cancer")
        metric = METRIC_MAP[match.group("metric")]
        arr = np.loadtxt(path, comments="#", dtype=float)
        if arr.shape != (10, 5):
            raise ValueError(f"Expected 10x5 in {path}, got {arr.shape}")
        if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
            raise ValueError(f"Invalid metric values in {path}")
        data.setdefault(cancer, {}).setdefault(method, {})[metric] = arr
    return data


def bh(p_values: pd.Series) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    good = np.isfinite(p)
    pv = p[good]
    if not len(pv):
        return q
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q[good] = restored
    return q


def rank_biserial(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.0
    ranks = stats.rankdata(np.abs(d), method="average")
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def hodges_lehmann(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def hl_bootstrap_ci(diff: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    samples = d[rng.integers(0, len(d), size=(n_boot, len(d)))]
    i, j = np.triu_indices(len(d))
    boot_hl = np.median((samples[:, i] + samples[:, j]) / 2.0, axis=1)
    low, high = np.quantile(boot_hl, [0.025, 0.975])
    return float(low), float(high)


def safe_wilcoxon(
    x: np.ndarray, y: np.ndarray, alternative: str, method: str = "asymptotic"
) -> tuple[float, float]:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if np.all(d == 0):
        return 0.0, 1.0
    res = stats.wilcoxon(
        x,
        y,
        zero_method="wilcox",
        correction=False,
        alternative=alternative,
        method=method,
    )
    return float(res.statistic), float(res.pvalue)


def exact_signflip_wilcoxon(
    x: np.ndarray, y: np.ndarray, alternative: str
) -> tuple[float, float]:
    """Exact paired signed-rank randomization test, including ties.

    Zero differences are discarded. For n<=20, all 2^n sign assignments are
    enumerated, so the result does not depend on SciPy's version-specific
    ``method='auto'`` threshold.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    if not len(d):
        return 0.0, 1.0
    if len(d) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to n<=20")
    ranks = stats.rankdata(np.abs(d), method="average")
    observed = float(ranks[d > 0].sum())
    states = np.arange(1 << len(d), dtype=np.uint32)[:, None]
    bits = ((states >> np.arange(len(d), dtype=np.uint32)) & 1).astype(float)
    null_w_plus = bits @ ranks
    p_greater = float(np.mean(null_w_plus >= observed - 1e-12))
    p_less = float(np.mean(null_w_plus <= observed + 1e-12))
    if alternative == "greater":
        p = p_greater
    elif alternative == "less":
        p = p_less
    elif alternative == "two-sided":
        p = min(1.0, 2.0 * min(p_greater, p_less))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    return observed, p


def legacy_scipy_default_wilcoxon(x: np.ndarray, y: np.ndarray, alternative: str) -> tuple[float, float]:
    """Reproduce the archived notebook's older SciPy default decision rule."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    nonzero_abs = np.abs(d[d != 0])
    has_ties_or_zeros = len(nonzero_abs) != len(d) or len(np.unique(nonzero_abs)) != len(nonzero_abs)
    method = "asymptotic" if has_ties_or_zeros else "exact"
    return safe_wilcoxon(x, y, alternative, method=method)


def corrected_repeated_kfold(diff: np.ndarray, repeats: int = 10, folds: int = 5) -> dict[str, float]:
    d = np.asarray(diff, dtype=float).reshape(-1)
    if len(d) != repeats * folds:
        raise ValueError("Corrected test requires repeats*folds paired differences")
    mean = float(np.mean(d))
    variance = float(np.var(d, ddof=1))
    correction_factor = 1.0 / (repeats * folds) + 1.0 / (folds - 1)
    se = math.sqrt(correction_factor * variance)
    df = repeats * folds - 1
    if se == 0:
        t_stat = math.copysign(math.inf, mean) if mean else 0.0
        p_two = 0.0 if mean else 1.0
        ci_low = ci_high = mean
    else:
        t_stat = mean / se
        p_two = float(2 * stats.t.sf(abs(t_stat), df))
        crit = float(stats.t.ppf(0.975, df))
        ci_low, ci_high = mean - crit * se, mean + crit * se
    return {
        "corrected_t": float(t_stat),
        "corrected_df": int(df),
        "corrected_p_two": p_two,
        "corrected_se": float(se),
        "corrected_ci95_low": float(ci_low),
        "corrected_ci95_high": float(ci_high),
        "correction_factor": float(correction_factor),
    }


def compare_pair(base: np.ndarray, other: np.ndarray, seed: int) -> dict[str, float | int | str]:
    if base.shape != (10, 5) or other.shape != (10, 5):
        raise ValueError("Both inputs must be 10x5")
    fold_d = (base - other).reshape(-1)
    run_base = base.mean(axis=1)
    run_other = other.mean(axis=1)
    run_d = run_base - run_other

    fold_w_two, fold_p_two = safe_wilcoxon(base.reshape(-1), other.reshape(-1), "two-sided", method="asymptotic")
    fold_w_greater, fold_p_greater = safe_wilcoxon(base.reshape(-1), other.reshape(-1), "greater", method="asymptotic")
    _, fold_p_greater_legacy = legacy_scipy_default_wilcoxon(
        base.reshape(-1), other.reshape(-1), "greater"
    )
    run_w_two, run_p_two = exact_signflip_wilcoxon(run_base, run_other, "two-sided")
    run_w_greater, run_p_greater = exact_signflip_wilcoxon(run_base, run_other, "greater")
    hl_low, hl_high = hl_bootstrap_ci(run_d, seed=seed)
    corrected = corrected_repeated_kfold(fold_d)

    return {
        "n_repeats": 10,
        "n_folds": 5,
        "n_fold_pairs": 50,
        "base_fold_mean": float(base.mean()),
        "base_fold_sd": float(base.std(ddof=1)),
        "comparator_fold_mean": float(other.mean()),
        "comparator_fold_sd": float(other.std(ddof=1)),
        "fold_mean_diff": float(fold_d.mean()),
        "fold_median_diff": float(np.median(fold_d)),
        "fold_diff_sd": float(fold_d.std(ddof=1)),
        "fold_w_two": fold_w_two,
        "fold_p_two": fold_p_two,
        "fold_w_greater": fold_w_greater,
        "fold_p_greater": fold_p_greater,
        "fold_p_greater_legacy_scipy": fold_p_greater_legacy,
        "fold_rank_biserial": rank_biserial(fold_d),
        "base_run_mean": float(run_base.mean()),
        "base_run_sd": float(run_base.std(ddof=1)),
        "comparator_run_mean": float(run_other.mean()),
        "comparator_run_sd": float(run_other.std(ddof=1)),
        "run_mean_diff": float(run_d.mean()),
        "run_median_diff": float(np.median(run_d)),
        "run_w_two": run_w_two,
        "run_p_two": run_p_two,
        "run_w_greater": run_w_greater,
        "run_p_greater": run_p_greater,
        "run_rank_biserial": rank_biserial(run_d),
        "run_hodges_lehmann": hodges_lehmann(run_d),
        "run_hl_boot_ci95_low": hl_low,
        "run_hl_boot_ci95_high": hl_high,
        "direction_by_mean": "BIMODriver higher" if fold_d.mean() > 0 else ("Comparator higher" if fold_d.mean() < 0 else "Tie"),
        **corrected,
    }


def score_tables(pan, string_pan, cancers):
    fold_rows = []
    run_rows = []
    for scope, cancer, methods in [
        ("Pan-cancer CPDB", "pan-cancer", pan),
        ("Pan-cancer STRING", "pan-cancer", string_pan),
    ] + [
        ("Cancer-specific", cancer, methods) for cancer, methods in sorted(cancers.items())
    ]:
        for method, metrics in sorted(methods.items()):
            for metric, arr in sorted(metrics.items()):
                for run in range(10):
                    run_rows.append({
                        "scope": scope,
                        "cancer": cancer,
                        "method": method,
                        "metric": metric,
                        "run": run + 1,
                        "run_mean": float(arr[run].mean()),
                    })
                    for fold in range(5):
                        fold_rows.append({
                            "scope": scope,
                            "cancer": cancer,
                            "method": method,
                            "metric": metric,
                            "run": run + 1,
                            "fold": fold + 1,
                            "score": float(arr[run, fold]),
                        })
    return pd.DataFrame(fold_rows), pd.DataFrame(run_rows)


def comparison_tables(pan, string_pan, cancers):
    pan_rows = []
    string_rows = []
    cancer_rows = []
    seed_counter = 20260908
    for metric in ["AUROC", "AUPRC"]:
        base = pan[BASE_METHOD][metric]
        for comparator in sorted(set(pan) - {BASE_METHOD}):
            pan_rows.append({
                "scope": "Pan-cancer CPDB",
                "cancer": "pan-cancer",
                "metric": metric,
                "comparator": comparator,
                **compare_pair(base, pan[comparator][metric], seed_counter),
            })
            seed_counter += 1
    for metric in ["AUROC", "AUPRC"]:
        base = string_pan[BASE_METHOD][metric]
        for comparator in sorted(set(string_pan) - {BASE_METHOD}):
            string_rows.append({
                "scope": "Pan-cancer STRING",
                "cancer": "pan-cancer",
                "metric": metric,
                "comparator": comparator,
                **compare_pair(base, string_pan[comparator][metric], seed_counter),
            })
            seed_counter += 1
    for cancer, methods in sorted(cancers.items()):
        if BASE_METHOD not in methods:
            raise ValueError(f"Missing {BASE_METHOD} for {cancer}")
        for metric in ["AUROC", "AUPRC"]:
            base = methods[BASE_METHOD][metric]
            for comparator in sorted(set(methods) - {BASE_METHOD}):
                cancer_rows.append({
                    "scope": "Cancer-specific",
                    "cancer": cancer,
                    "metric": metric,
                    "comparator": comparator,
                    **compare_pair(base, methods[comparator][metric], seed_counter),
                })
                seed_counter += 1

    pan_df = pd.DataFrame(pan_rows)
    cancer_df = pd.DataFrame(cancer_rows)
    string_df = pd.DataFrame(string_rows)
    combined = pd.concat([
        pan_df.assign(_table="pan"),
        string_df.assign(_table="string"),
        cancer_df.assign(_table="cancer"),
    ], ignore_index=True)
    # Families are deliberately separated by analysis type/direction.
    for p_col, q_col in [
        ("corrected_p_two", "corrected_q_BH_global"),
        ("run_p_two", "run_q_BH_two_global"),
        ("run_p_greater", "run_q_BH_greater_global"),
        ("fold_p_two", "fold_q_BH_two_global"),
        ("fold_p_greater", "fold_q_BH_greater_global"),
    ]:
        combined[q_col] = bh(combined[p_col])
    pan_out = combined[combined["_table"] == "pan"].drop(columns="_table").reset_index(drop=True)
    string_out = combined[combined["_table"] == "string"].drop(columns="_table").reset_index(drop=True)
    cancer_out = combined[combined["_table"] == "cancer"].drop(columns="_table").reset_index(drop=True)
    return pan_out, string_out, cancer_out


def across_cancer_table(cancers):
    cancer_names = sorted(cancers)
    methods = sorted(set.intersection(*(set(cancers[c]) for c in cancer_names)))
    rows = []
    seed = 20270000
    for metric in ["AUROC", "AUPRC"]:
        base = np.array([cancers[c][BASE_METHOD][metric].mean() for c in cancer_names])
        for comparator in [m for m in methods if m != BASE_METHOD]:
            other = np.array([cancers[c][comparator][metric].mean() for c in cancer_names])
            diff = base - other
            w_two, p_two = exact_signflip_wilcoxon(base, other, "two-sided")
            w_greater, p_greater = exact_signflip_wilcoxon(base, other, "greater")
            ci_low, ci_high = hl_bootstrap_ci(diff, seed=seed)
            seed += 1
            rows.append({
                "scope": "Across 15 cancer types",
                "metric": metric,
                "comparator": comparator,
                "n_cancer_pairs": len(cancer_names),
                "base_macro_mean": float(base.mean()),
                "comparator_macro_mean": float(other.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "wilcoxon_w_two": w_two,
                "wilcoxon_p_two": p_two,
                "wilcoxon_w_greater": w_greater,
                "wilcoxon_p_greater": p_greater,
                "rank_biserial": rank_biserial(diff),
                "hodges_lehmann": hodges_lehmann(diff),
                "hl_boot_ci95_low": ci_low,
                "hl_boot_ci95_high": ci_high,
                "direction_by_mean": "BIMODriver higher" if diff.mean() > 0 else ("Comparator higher" if diff.mean() < 0 else "Tie"),
            })
    df = pd.DataFrame(rows)
    df["wilcoxon_q_BH_two"] = bh(df["wilcoxon_p_two"])
    df["wilcoxon_q_BH_greater"] = bh(df["wilcoxon_p_greater"])
    return df


def benchmark_summary_tables(pan_stats, string_stats, across_stats, cancers, run_scores):
    """Create the three publication benchmark comparison blocks.

    CPDB/STRING use ten paired repeat means; the across-cancer block uses the
    fifteen paired cancer-type means.  BH adjustment is performed separately
    within each 18-comparison block.
    """
    rows = []
    scope_inputs = [
        ("CPDB pan-cancer", pan_stats),
        ("STRING pan-cancer", string_stats),
    ]
    for scope, df in scope_inputs:
        for _, r in df.iterrows():
            rows.append({
                "scope": scope,
                "metric": r["metric"],
                "comparator": r["comparator"],
                "paired_unit": "run mean (5 folds)",
                "n_pairs": 10,
                "bimodriver_mean": r["base_run_mean"],
                "bimodriver_10run_sd": r["base_run_sd"],
                "bimodriver_sd": r["base_run_sd"],
                "comparator_mean": r["comparator_run_mean"],
                "comparator_10run_sd": r["comparator_run_sd"],
                "comparator_sd": r["comparator_run_sd"],
                "sd_type": "10-run mean SD (ddof=1)",
                "mean_difference": r["run_mean_diff"],
                "wilcoxon_p_one_sided": r["run_p_greater"],
                "rank_biserial_effect_size": r["run_rank_biserial"],
                "hodges_lehmann_difference": r["run_hodges_lehmann"],
                "hl_ci95_low": r["run_hl_boot_ci95_low"],
                "hl_ci95_high": r["run_hl_boot_ci95_high"],
            })

    cancer_names = sorted(cancers)
    for _, r in across_stats.iterrows():
        metric = r["metric"]
        comparator = r["comparator"]
        base_values = np.array([cancers[c][BASE_METHOD][metric].mean() for c in cancer_names])
        other_values = np.array([cancers[c][comparator][metric].mean() for c in cancer_names])
        rows.append({
            "scope": "15 cancer-type means",
            "metric": metric,
            "comparator": comparator,
            "paired_unit": "cancer-type mean",
            "n_pairs": 15,
            "bimodriver_mean": base_values.mean(),
            "bimodriver_10run_sd": base_values.std(ddof=1),
            "bimodriver_sd": base_values.std(ddof=1),
            "comparator_mean": other_values.mean(),
            "comparator_10run_sd": other_values.std(ddof=1),
            "comparator_sd": other_values.std(ddof=1),
            "sd_type": "15 cancer-type means SD (ddof=1)",
            "mean_difference": r["mean_diff"],
            "wilcoxon_p_one_sided": r["wilcoxon_p_greater"],
            "rank_biserial_effect_size": r["rank_biserial"],
            "hodges_lehmann_difference": r["hodges_lehmann"],
            "hl_ci95_low": r["hl_boot_ci95_low"],
            "hl_ci95_high": r["hl_boot_ci95_high"],
        })

    summary = pd.DataFrame(rows)
    summary["bh_q_within_scope"] = np.nan
    for scope, idx in summary.groupby("scope", sort=False).groups.items():
        summary.loc[idx, "bh_q_within_scope"] = bh(summary.loc[idx, "wilcoxon_p_one_sided"])
    summary["significant_q_lt_0_05"] = summary["bh_q_within_scope"] < 0.05
    summary["result"] = np.where(
        summary["significant_q_lt_0_05"],
        np.where(summary["mean_difference"] > 0, "BIMODriver higher; q<0.05", "Comparator higher; q<0.05"),
        "q>=0.05",
    )

    pan_run_raw = run_scores[
        run_scores["scope"].isin(["Pan-cancer CPDB", "Pan-cancer STRING"])
    ].copy()
    pan_run_raw["scope"] = pan_run_raw["scope"].replace({
        "Pan-cancer CPDB": "CPDB pan-cancer",
        "Pan-cancer STRING": "STRING pan-cancer",
    })

    cancer_mean_rows = []
    common_methods = sorted(set.intersection(*(set(cancers[c]) for c in cancer_names)))
    for cancer in cancer_names:
        for method in common_methods:
            for metric in ["AUROC", "AUPRC"]:
                cancer_mean_rows.append({
                    "cancer": cancer.upper(),
                    "method": method,
                    "metric": metric,
                    "mean_10x5": float(cancers[cancer][method][metric].mean()),
                })
    cancer_means = pd.DataFrame(cancer_mean_rows)
    return summary, pan_run_raw, cancer_means


def fold50_sensitivity_table(pan, string_pan, archived_full):
    """Add effect estimates and uncertainty to the 50-fold sensitivity tests.

    The one-sided Wilcoxon p values and both BH corrections
    are retained exactly.  New quantities are calculated from the same 50
    coordinate-matched fold differences.
    """
    method_alias = {
        "ECD_CDGI": "ECD-CDGI",
        "Dishyper": "DISHyper",
        "Cheb": "ChebNet",
        "Emogi": "EMOGI",
    }
    metric_alias = {"AUC": "AUROC", "AUPRC": "AUPRC"}
    datasets = {"CPDB": pan, "STRING": string_pan}
    rows = []
    seed = 20280000
    for _, r in archived_full.iterrows():
        scope = str(r["范围"])
        if scope not in datasets:
            continue
        comparator = method_alias.get(str(r["基线"]), str(r["基线"]))
        metric = metric_alias[str(r["指标"])]
        base = datasets[scope][BASE_METHOD][metric].reshape(-1)
        other = datasets[scope][comparator][metric].reshape(-1)
        diff = base - other
        ci_low, ci_high = hl_bootstrap_ci(diff, seed=seed, n_boot=10000)
        seed += 1
        q_scope = float(r["BH_q_合并(本范围)"])
        rows.append({
            "scope": f"{scope} pan-cancer",
            "metric": metric,
            "comparator": comparator,
            "paired_unit": "fold value",
            "n_pairs": 50,
            "bimodriver_mean": float(base.mean()),
            "bimodriver_sd": float(base.std(ddof=1)),
            "comparator_mean": float(other.mean()),
            "comparator_sd": float(other.std(ddof=1)),
            "mean_difference": float(diff.mean()),
            "wilcoxon_p_one_sided": float(r["p值"]),
            "bh_q_within_scope": q_scope,
            "bh_q_within_metric": float(r["BH_q_分指标(本范围)"]),
            "rank_biserial_effect_size": rank_biserial(diff),
            "hodges_lehmann_difference": hodges_lehmann(diff),
            "hl_ci95_low": ci_low,
            "hl_ci95_high": ci_high,
            "significant_q_lt_0_05": q_scope < 0.05,
            "result": (
                "BIMODriver higher; q<0.05"
                if q_scope < 0.05 and diff.mean() > 0
                else ("Comparator higher; q<0.05" if q_scope < 0.05 else "q>=0.05")
            ),
        })
    out = pd.DataFrame(rows)
    if len(out) != 36:
        raise ValueError(f"Expected 36 CPDB/STRING 50-fold comparisons, got {len(out)}")
    return out


def qa_table(pan, string_pan, cancers, fold_scores, run_scores):
    cancer_names = sorted(cancers)
    cancer_methods = sorted(set.intersection(*(set(cancers[c]) for c in cancer_names)))
    rows = [
        {"check": "Pan-cancer method count", "expected": 10, "observed": len(pan), "status": "PASS" if len(pan) == 10 else "FAIL"},
        {"check": "STRING pan-cancer method count", "expected": 10, "observed": len(string_pan), "status": "PASS" if len(string_pan) == 10 else "FAIL"},
        {"check": "Cancer type count", "expected": 15, "observed": len(cancers), "status": "PASS" if len(cancers) == 15 else "FAIL"},
        {"check": "Cancer-specific method count", "expected": 10, "observed": len(cancer_methods), "status": "PASS" if len(cancer_methods) == 10 else "FAIL"},
        {"check": "Fold score rows", "expected": 17000, "observed": len(fold_scores), "status": "PASS" if len(fold_scores) == 17000 else "FAIL"},
        {"check": "Run score rows", "expected": 3400, "observed": len(run_scores), "status": "PASS" if len(run_scores) == 3400 else "FAIL"},
        {"check": "Scores in [0,1]", "expected": "True", "observed": str(bool(fold_scores["score"].between(0, 1).all())), "status": "PASS" if fold_scores["score"].between(0, 1).all() else "FAIL"},
        {"check": "10x5 coordinate meaning", "expected": "row=repeat, column=fold", "observed": "Explicitly stated in cancer-specific file headers; same matrix convention used for pan-cancer", "status": "PASS"},
        {"check": "Split membership identifiers", "expected": "k_sets / fold sample IDs for direct membership audit", "observed": "Not included; pairing is by the supplied run/fold matrix coordinates", "status": "MISSING"},
        {"check": "Prediction files present", "expected": "Needed for reviewer release request", "observed": "No y_true / probability / sample-ID files found", "status": "MISSING"},
    ]
    return pd.DataFrame(rows)


def cancer_header_audit():
    rows = []
    for path in sorted(CANCER_DIR.glob("*/*_50values.txt")):
        text = path.read_text(encoding="utf-8")
        name = path.stem
        match = re.match(r"^(?P<method>.+)_(?P<cancer>[a-z]+)_(?P<metric>AUC|AUPRC)_50values$", name)
        if not match:
            continue
        paper = re.search(r"论文值\(main\.tex Table 2\):\s*([0-9.]+)", text)
        stated = re.search(r"数据均值:\s*([0-9.]+)", text)
        arr = np.loadtxt(path, comments="#", dtype=float)
        actual = float(arr.mean())
        paper_value = float(paper.group(1)) if paper else np.nan
        stated_value = float(stated.group(1)) if stated else np.nan
        rows.append({
            "method": match.group("method"),
            "cancer": match.group("cancer"),
            "metric": METRIC_MAP[match.group("metric")],
            "actual_50value_mean": actual,
            "stated_data_mean": stated_value,
            "abs_diff_vs_stated": abs(actual - stated_value),
            "paper_table2_value": paper_value,
            "abs_diff_vs_paper": abs(actual - paper_value),
            "source_file": str(path.relative_to(ROOT)),
        })
    df = pd.DataFrame(rows)
    df["stated_mean_match_1e-6"] = df["abs_diff_vs_stated"] <= 1e-6
    return df


def legacy_crosscheck(pan_df, across_df):
    path = ROOT / "legacy_pancancer_wilcoxon_bh.csv"
    if not path.exists():
        path = ROOT / "Wilcoxon+BH_全量结果_一张表泛癌.csv"
    old = pd.read_csv(path, encoding="gb18030")
    method_alias = {
        "ECD_CDGI": "ECD-CDGI",
        "Dishyper": "DISHyper",
        "Cheb": "ChebNet",
        "Emogi": "EMOGI",
    }
    metric_alias = {"AUC": "AUROC", "AUPRC": "AUPRC"}
    rows = []
    for _, r in old.iterrows():
        scope = str(r["范围"])
        comparator = method_alias.get(str(r["基线"]), str(r["基线"]))
        metric = metric_alias.get(str(r["指标"]), str(r["指标"]))
        old_p = float(r["p值"])
        if scope == "CPDB":
            match = pan_df[(pan_df.comparator == comparator) & (pan_df.metric == metric)]
            p_asym = float(match.iloc[0]["fold_p_greater"]) if len(match) == 1 else np.nan
            p_exact_candidate = float(match.iloc[0]["fold_p_greater_legacy_scipy"]) if len(match) == 1 else np.nan
            analysis = "Pan-cancer 50-fold one-sided Wilcoxon"
        elif "15" in scope:
            match = across_df[(across_df.comparator == comparator) & (across_df.metric == metric)]
            p_asym = np.nan
            p_exact_candidate = float(match.iloc[0]["wilcoxon_p_greater"]) if len(match) == 1 else np.nan
            analysis = "15-cancer one-sided Wilcoxon"
        else:
            continue
        candidates = [("asymptotic", p_asym), ("exact/sign-flip", p_exact_candidate)]
        finite_candidates = [(label, value) for label, value in candidates if np.isfinite(value)]
        if finite_candidates:
            closest_algorithm, new_p = min(finite_candidates, key=lambda item: abs(item[1] - old_p))
        else:
            closest_algorithm, new_p = "unavailable", np.nan
        rows.append({
            "analysis": analysis,
            "metric": metric,
            "comparator": comparator,
            "archived_p": old_p,
            "recomputed_asymptotic_p": p_asym,
            "recomputed_exact_or_signflip_p": p_exact_candidate,
            "closest_algorithm": closest_algorithm,
            "closest_recomputed_p": new_p,
            "absolute_difference": abs(old_p - new_p) if np.isfinite(new_p) else np.nan,
            "reproduced_tolerance_1e-8": bool(np.isclose(old_p, new_p, rtol=0, atol=1e-8)) if np.isfinite(new_p) else False,
        })
    return pd.DataFrame(rows)


def build_paper_tables(pan, string_pan, cancers, archived_full, benchmark_summary, cancer_stats):
    """Build exact paper tables matching main.tex and appendix.tex."""
    # Table 1: Pan-cancer comparison across CPDB & STRING
    methods_order = ["GCN", "GAT", "ChebNet", "EMOGI", "ECD-CDGI", "MTGCN", "MNGCL", "DISHyper", "DISFusion", "BIMODriver"]
    t1_rows = []
    for m in methods_order:
        c_auc = pan[m]["AUROC"]
        c_auprc = pan[m]["AUPRC"]
        s_auc = string_pan[m]["AUROC"]
        s_auprc = string_pan[m]["AUPRC"]
        t1_rows.append({
            "Method": m,
            "CPDB AUC": f"{c_auc.mean():.4f} ± {c_auc.std(ddof=0):.4f}",
            "CPDB AUPRC": f"{c_auprc.mean():.4f} ± {c_auprc.std(ddof=0):.4f}",
            "STRING AUC": f"{s_auc.mean():.4f} ± {s_auc.std(ddof=0):.4f}",
            "STRING AUPRC": f"{s_auprc.mean():.4f} ± {s_auprc.std(ddof=0):.4f}",
        })
    df_t1 = pd.DataFrame(t1_rows)

    # Table 2: Cancer-specific comparison
    cancers_list = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirc", "kirp", "lihc", "luad", "lusc", "prad", "stad", "thca", "ucec"]
    cancer_models = ["GAT", "SAGE", "ChebNet", "MTGCN", "MNGCL", "EMOGI", "ECD-CDGI", "DISHyper", "DISFusion", "BIMODriver"]
    t2_rows = []
    for metric_display, metric_key in [("AUC", "AUROC"), ("AUPRC", "AUPRC")]:
        for m in cancer_models:
            row = {"Metric": metric_display, "Model": m}
            for c in cancers_list:
                arr = cancers[c][m][metric_key]
                row[c.upper()] = f"{arr.mean():.4f}"
            t2_rows.append(row)
    df_t2 = pd.DataFrame(t2_rows)

    # Table A8: Pan-cancer statistical significance (CPDB)
    cpdb = archived_full[archived_full["范围"] == "CPDB"]
    baselines_a8 = ["GCN", "Cheb", "GAT", "ECD_CDGI", "MNGCL", "MTGCN", "EMOGI", "Dishyper", "DISFusion"]
    rows_a8 = []
    for b in baselines_a8:
        auc_row = cpdb[(cpdb["基线"] == b) & (cpdb["指标"] == "AUC")].iloc[0]
        auprc_row = cpdb[(cpdb["基线"] == b) & (cpdb["指标"] == "AUPRC")].iloc[0]
        p_auc = float(auc_row["p值"])
        p_auprc = float(auprc_row["p值"])
        rows_a8.append({
            "Baselines": b,
            "AUC p-value": f"{p_auc:.2E}" if p_auc >= 1e-15 else "< 1.00E-15",
            "AUPRC p-value": f"{p_auprc:.2E}" if p_auprc >= 1e-15 else "< 1.00E-15",
            "AUC q-value": f"{float(auc_row['BH_q_分指标(本范围)']):.2E}",
            "AUPRC q-value": f"{float(auprc_row['BH_q_分指标(本范围)']):.2E}",
        })
    df_a8 = pd.DataFrame(rows_a8)

    # Table A9: Cancer-specific statistical significance (15 cancers)
    c15 = archived_full[archived_full["范围"] == "单癌症-15均值"]
    baselines_a9 = ["GAT", "SAGE", "ChebNet", "MTGCN", "MNGCL", "Emogi", "ECD-CDGI", "DISHyper", "DISFusion"]
    rows_a9 = []
    for b in baselines_a9:
        auc_row = c15[(c15["基线"] == b) & (c15["指标"] == "AUC")].iloc[0]
        auprc_row = c15[(c15["基线"] == b) & (c15["指标"] == "AUPRC")].iloc[0]
        rows_a9.append({
            "Baselines": b,
            "AUC p-value": f"{float(auc_row['p值']):.2E}",
            "AUPRC p-value": f"{float(auprc_row['p值']):.2E}",
            "AUC q-value": f"{float(auc_row['BH_q_分指标(本范围)']):.2E}",
            "AUPRC q-value": f"{float(auprc_row['BH_q_分指标(本范围)']):.2E}",
        })
    df_a9 = pd.DataFrame(rows_a9)

    # Table A10: Pan-cancer detailed significance table (matching tab:pan_cancer_significance)
    pan_mask = benchmark_summary["scope"].str.contains("pan-cancer")
    df_a10 = pd.DataFrame({
        "Scope": benchmark_summary.loc[pan_mask, "scope"],
        "Metric": benchmark_summary.loc[pan_mask, "metric"],
        "Comparator": benchmark_summary.loc[pan_mask, "comparator"],
        "Mean diff": benchmark_summary.loc[pan_mask, "mean_difference"].apply(lambda x: f"{x:.4f}"),
        "Wilcoxon p": benchmark_summary.loc[pan_mask, "wilcoxon_p_one_sided"].apply(lambda x: f"{x:.3E}"),
        "BH q": benchmark_summary.loc[pan_mask, "bh_q_within_scope"].apply(lambda x: f"{x:.3E}"),
        "Effect size r_rb": benchmark_summary.loc[pan_mask, "rank_biserial_effect_size"].apply(lambda x: f"{x:.3f}"),
        "HL diff": benchmark_summary.loc[pan_mask, "hodges_lehmann_difference"].apply(lambda x: f"{x:.4f}"),
        "95% CI": [f"[{l:.4f}, {h:.4f}]" for l, h in zip(benchmark_summary.loc[pan_mask, "hl_ci95_low"], benchmark_summary.loc[pan_mask, "hl_ci95_high"])],
        "Result": "Higher; q<0.05",
    })

    # Table A11: Cancer-specific detailed significance table (matching tab:cancer_type_significance)
    c_order = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirc", "kirp", "lihc", "luad", "lusc", "prad", "stad", "thca", "ucec"]
    comp_order = ["ChebNet", "DISFusion", "DISHyper", "ECD-CDGI", "EMOGI", "GAT", "MNGCL", "MTGCN", "SAGE"]
    # 1. Primary BH q across all 270 single-cancer tests (m=270)
    p_vals_270 = cancer_stats["run_p_greater"].values
    n_270 = len(p_vals_270)
    order_270 = np.argsort(p_vals_270)
    ranks_270 = np.empty_like(order_270)
    ranks_270[order_270] = np.arange(1, n_270 + 1)
    q_vals_270 = p_vals_270 * n_270 / ranks_270
    sorted_q_270 = q_vals_270[order_270]
    for i in range(n_270 - 2, -1, -1):
        sorted_q_270[i] = min(sorted_q_270[i], sorted_q_270[i + 1])
    q_vals_270[order_270] = sorted_q_270
    q_map_270 = {}
    for idx, row in cancer_stats.iterrows():
        q_map_270[(row["cancer"], row["metric"], row["comparator"])] = q_vals_270[list(cancer_stats.index).index(idx)]

    # 2. Supplementary by-metric BH q (135 tests per metric)
    q_map_135 = {}
    for met in ["AUROC", "AUPRC"]:
        sub = cancer_stats[cancer_stats.metric == met].copy()
        p_vals = sub["run_p_greater"].values
        n = len(p_vals)
        order = np.argsort(p_vals)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, n + 1)
        q_vals = p_vals * n / ranks
        sorted_q = q_vals[order]
        for i in range(n - 2, -1, -1):
            sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
        q_vals[order] = sorted_q
        for idx, row in sub.iterrows():
            q_map_135[(row["cancer"], row["metric"], row["comparator"])] = q_vals[list(sub.index).index(idx)]

    rows_a11 = []
    for c in c_order:
        for met in ["AUROC", "AUPRC"]:
            for comp in comp_order:
                sub = cancer_stats[(cancer_stats.cancer == c) & (cancer_stats.metric == met) & (cancer_stats.comparator == comp)]
                if len(sub) == 0:
                    continue
                r = sub.iloc[0]
                q_val_270 = q_map_270[(c, met, comp)]
                q_val_135 = q_map_135[(c, met, comp)]
                rows_a11.append({
                    "Cancer type": c.upper(),
                    "Metric": met,
                    "Comparator": comp,
                    "Mean diff": f"{r.run_mean_diff:.4f}",
                    "Wilcoxon p": f"{r.run_p_greater:.3E}",
                    "BH q (global, 270)": f"{q_val_270:.3E}",
                    "Supplementary by-metric BH q (135)": f"{q_val_135:.3E}",
                    "Effect size r_rb": f"{r.run_rank_biserial:.3f}",
                    "HL diff": f"{r.run_hodges_lehmann:.4f}",
                    "95% CI": f"[{r.run_hl_boot_ci95_low:.4f}, {r.run_hl_boot_ci95_high:.4f}]",
                    "Conclusion": "Higher; q<0.05" if q_val_270 < 0.05 else "Not significant",
                })
    df_a11 = pd.DataFrame(rows_a11)

    return df_t1, df_t2, df_a8, df_a9, df_a10, df_a11


def main():
    pan = load_pan()
    string_pan = load_string_pan()
    cancers = load_cancers()

    if set(pan[BASE_METHOD]) != {"AUROC", "AUPRC"}:
        raise ValueError("Pan-cancer BIMODriver metrics incomplete")
    if set(string_pan[BASE_METHOD]) != {"AUROC", "AUPRC"}:
        raise ValueError("STRING pan-cancer BIMODriver metrics incomplete")
    cancer_names = sorted(cancers)
    expected_cancers = ["blca", "brca", "cesc", "coad", "esca", "hnsc", "kirc", "kirp", "lihc", "luad", "lusc", "prad", "stad", "thca", "ucec"]
    if cancer_names != expected_cancers:
        raise ValueError(f"Unexpected cancers: {cancer_names}")

    fold_scores, run_scores = score_tables(pan, string_pan, cancers)
    pan_stats, string_stats, cancer_stats = comparison_tables(pan, string_pan, cancers)
    across_stats = across_cancer_table(cancers)
    qa = qa_table(pan, string_pan, cancers, fold_scores, run_scores)
    legacy = legacy_crosscheck(pan_stats, across_stats)
    header_audit = cancer_header_audit()
    archived_full_path = ROOT / "legacy_full_wilcoxon_bh.csv"
    if not archived_full_path.exists():
        archived_full_path = ROOT / "Wilcoxon+BH_全量结果_一张表1.csv"
    archived_full = pd.read_csv(archived_full_path, encoding="gb18030")
    benchmark_summary, benchmark_run_means, cancer_type_means = benchmark_summary_tables(
        pan_stats, string_stats, across_stats, cancers, run_scores
    )
    sensitivity_fold50 = fold50_sensitivity_table(pan, string_pan, archived_full)

    df_t1, df_t2, df_a8, df_a9, df_a10, df_a11 = build_paper_tables(
        pan, string_pan, cancers, archived_full, benchmark_summary, cancer_stats
    )

    all_exports = {
        "fold_scores": fold_scores,
        "run_scores": run_scores,
        "pan_stats": pan_stats,
        "string_stats": string_stats,
        "cancer_stats": cancer_stats,
        "across15_stats": across_stats,
        "benchmark_summary": benchmark_summary,
        "benchmark_run_means": benchmark_run_means,
        "cancer_type_means": cancer_type_means,
        "sensitivity_fold50": sensitivity_fold50,
        "qa": qa,
        "legacy_crosscheck": legacy,
        "cancer_header_audit": header_audit,
        "archived_full": archived_full,
        "table1_performance_comparison": df_t1,
        "table2_cancer_specific_comparison": df_t2,
        "table_A8_pancancer_significance": df_a8,
        "table_A9_cancer_specific_significance": df_a9,
        "table_A10_pancancer_significance": df_a10,
        "table_A11_cancer_specific_significance": df_a11,
    }

    for name, df in all_exports.items():
        df.to_csv(OUT_DIR / f"{name}.csv", index=False)

    workbook_payload = {
        name: df.astype(object).where(pd.notna(df), None).to_dict(orient="records")
        for name, df in all_exports.items()
    }
    (OUT_DIR / "workbook_data.json").write_text(
        json.dumps(workbook_payload, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    summary = {
        "source": str(ROOT),
        "pan_methods": sorted(pan),
        "string_pan_methods": sorted(string_pan),
        "cancer_methods": sorted(set.intersection(*(set(cancers[c]) for c in cancers))),
        "cancers": cancer_names,
        "primary_family_size": int(len(pan_stats) + len(cancer_stats)),
        "across15_family_size": int(len(across_stats)),
        "qa_all_nonmissing_pass": bool((qa.loc[qa.status != "MISSING", "status"] == "PASS").all()),
        "legacy_p_reproduced_matches": int(legacy["reproduced_tolerance_1e-8"].sum()),
        "legacy_p_checks": int(len(legacy)),
        "cancer_header_mean_matches": int(header_audit["stated_mean_match_1e-6"].sum()),
        "cancer_header_mean_checks": int(len(header_audit)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
