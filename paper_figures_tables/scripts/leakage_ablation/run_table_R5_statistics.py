#!/usr/bin/env python3
"""Statistical validation for Table R5: Label-Like Phrases Masking Analysis.

Evaluates keyword / phrase masking ablation on CPDB pan-cancer benchmark across:
1. Original Transductive BIMODriver (paper reference)
2. Masked Transductive (keyword-masked features)
3. Masked Inductive (keyword-masked features + strict inductive graph learning)

Statistical Protocol:
- Unit of pairing: 10 run-level means, each averaged across 5 folds (n=10).
- 4 Primary Comparisons:
    1. AUROC: Masked Transductive vs Original Transductive BIMODriver
    2. AUROC: Masked Inductive vs Original Transductive BIMODriver
    3. AUPRC: Masked Transductive vs Original Transductive BIMODriver
    4. AUPRC: Masked Inductive vs Original Transductive BIMODriver
- Hypothesis testing: Two-sided paired Wilcoxon signed-rank test.
- Multiple testing: Benjamini-Hochberg FDR correction across the m=4 test family.
- Effect size: Paired rank-biserial correlation (r_rb), sign consistent with Left minus Right.
- Location shift: Paired Hodges-Lehmann difference (median of Walsh averages).
- Confidence Interval: 95% bootstrap CI for HL difference (10,000 resamples, fixed seed).
- Standard deviation: ddof=0 for all descriptive statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from scipy import stats


def exact_signed_rank_p(diff: np.ndarray) -> float:
    """Exact two-sided sign-flip p-value handling ties."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0, atol=1e-15)]
    if not len(d):
        return 1.0
    if len(d) > 20:
        raise ValueError("Exact enumeration is limited to at most 20 nonzero pairs")
    ranks = stats.rankdata(np.abs(d), method="average")
    observed = float(ranks[d > 0].sum())
    null = np.asarray(
        [
            sum(rank for rank, positive in zip(ranks, signs) if positive)
            for signs in product((0, 1), repeat=len(d))
        ],
        dtype=float,
    )
    center = float(ranks.sum()) / 2.0
    return float(np.mean(np.abs(null - center) >= abs(observed - center) - 1e-12))


def rank_biserial(diff: np.ndarray) -> float:
    """Paired rank-biserial correlation: r_rb = (W_+ - W_-) / (W_+ + W_-)."""
    d = np.asarray(diff, dtype=float)
    d = d[~np.isclose(d, 0.0, atol=1e-15)]
    if not len(d):
        return 0.0
    ranks = stats.rankdata(np.abs(d), method="average")
    positive = float(ranks[d > 0].sum())
    negative = float(ranks[d < 0].sum())
    return (positive - negative) / (positive + negative)


def hodges_lehmann(diff: np.ndarray) -> float:
    """Hodges-Lehmann difference (median of Walsh averages)."""
    d = np.asarray(diff, dtype=float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2.0))


def bootstrap_hl_ci(
    diff: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 20260916,
) -> tuple[float, float]:
    """Percentile bootstrap CI for paired Hodges-Lehmann difference."""
    d = np.asarray(diff, dtype=float)
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(len(d))
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    samples = d[idx]
    hl_vals = np.median((samples[:, i] + samples[:, j]) / 2.0, axis=1)
    low, high = np.quantile(hl_vals, [0.025, 0.975])
    return float(low), float(high)


def bh_adjust(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def parse_bracket_matrix(text: str) -> np.ndarray:
    """Parse numpy array text formatted with brackets into 10x5 array."""
    cleaned = text.replace("[", " ").replace("]", " ").strip()
    vals = [float(x) for x in cleaned.split()]
    arr = np.array(vals, dtype=float).reshape(10, 5)
    return arr


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_data(data_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, str]]:
    """Load and validate all raw 10x5 matrices."""
    files = {
        "cpdb_pancancer": data_dir / "cpdb-pancancer.txt",
        "masked_auroc": data_dir / "pan-cancer_masked_auroc.txt",
        "masked_auprc": data_dir / "pan-cancer_masked_auprc.txt",
        "masked_ind_auroc": data_dir / "pan-cancer_masked_inductive_auroc.txt",
        "masked_ind_auprc": data_dir / "pan-cancer_masked_inductive_auprc.txt",
    }
    hashes = {}
    for k, p in files.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required data file: {p}")
        hashes[k] = compute_sha256(p)

    # 1. Parse Original Transductive BIMODriver from cpdb-pancancer.txt
    cpdb_text = files["cpdb_pancancer"].read_text(encoding="utf-8")
    auprc_match = re.search(r"AUPR:.*?\n(\[\[.*?\]\])", cpdb_text, re.DOTALL)
    auroc_match = re.search(r"AUC:.*?\n(\[\[.*?\]\])", cpdb_text, re.DOTALL)
    if not auprc_match or not auroc_match:
        raise ValueError("Could not parse AUC and AUPR matrices from cpdb-pancancer.txt")
    orig_auprc = parse_bracket_matrix(auprc_match.group(1))
    orig_auroc = parse_bracket_matrix(auroc_match.group(1))

    # 2. Parse Masked Transductive
    mt_auroc = np.loadtxt(files["masked_auroc"], dtype=float)
    mt_auprc = np.loadtxt(files["masked_auprc"], dtype=float)

    # 3. Parse Masked Inductive
    mi_auroc = np.loadtxt(files["masked_ind_auroc"], dtype=float)
    mi_auprc = np.loadtxt(files["masked_ind_auprc"], dtype=float)

    data = {
        "Original Transductive BIMODriver": {"AUROC": orig_auroc, "AUPRC": orig_auprc},
        "Masked Transductive": {"AUROC": mt_auroc, "AUPRC": mt_auprc},
        "Masked Inductive": {"AUROC": mi_auroc, "AUPRC": mi_auprc},
    }

    # Integrity & shape checks
    for mod_name, metrics in data.items():
        for met_name, arr in metrics.items():
            if arr.shape != (10, 5):
                raise ValueError(f"Expected shape (10, 5) for {mod_name} {met_name}, got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"Non-finite values found in {mod_name} {met_name}")
            if not np.all((arr >= 0.0) & (arr <= 1.0)):
                raise ValueError(f"Values outside [0, 1] found in {mod_name} {met_name}")

    return data, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Table R5 label-like phrases masking statistical tests")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data" / "label-like_phrases_masking")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "label_like_phrases_masking")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, hashes = load_data(args.data_dir)

    print("=" * 80)
    print("TABLE R5: LABEL-LIKE PHRASES MASKING STATISTICAL VALIDATION")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Bootstrap resamples: {args.bootstrap:,} | Base random seed: {args.seed}")
    print("-" * 80)

    # 1. Descriptive Summary
    desc_rows = []
    for model_name in ["Original Transductive BIMODriver", "Masked Transductive", "Masked Inductive"]:
        for metric in ["AUROC", "AUPRC"]:
            arr = data[model_name][metric]
            fold50_mean = float(arr.mean())
            fold50_sd = float(arr.std(ddof=0))
            run_means = arr.mean(axis=1)
            run10_mean = float(run_means.mean())
            run10_sd = float(run_means.std(ddof=0))

            desc_rows.append({
                "Model Configuration": model_name,
                "Metric": metric,
                "50-Fold Mean": fold50_mean,
                "50-Fold SD (ddof=0)": fold50_sd,
                "50-Fold Mean ± SD": f"{fold50_mean:.4f} ± {fold50_sd:.4f}",
                "10-Run Mean": run10_mean,
                "10-Run SD (ddof=0)": run10_sd,
                "10-Run Mean ± SD": f"{run10_mean:.4f} ± {run10_sd:.4f}",
            })
    df_desc = pd.DataFrame(desc_rows)

    # 2. Run-Level Paired Values Table
    run_table_rows = []
    for run in range(10):
        row = {"Run": run + 1}
        for model_name in ["Original Transductive BIMODriver", "Masked Transductive", "Masked Inductive"]:
            for metric in ["AUROC", "AUPRC"]:
                col_name = f"{model_name} ({metric})"
                row[col_name] = float(data[model_name][metric][run].mean())
        run_table_rows.append(row)
    df_runs = pd.DataFrame(run_table_rows)

    # 3. Four Comparisons
    # Left minus Right
    ref_name = "Original Transductive BIMODriver"
    comps = [
        ("AUROC", "Masked Transductive", ref_name),
        ("AUROC", "Masked Inductive", ref_name),
        ("AUPRC", "Masked Transductive", ref_name),
        ("AUPRC", "Masked Inductive", ref_name),
    ]

    stat_rows_full = []
    for metric, left_name, right_name in comps:
        left_arr = data[left_name][metric]
        right_arr = data[right_name][metric]

        left_run = left_arr.mean(axis=1)
        right_run = right_arr.mean(axis=1)
        diff = left_run - right_run  # Left minus Right

        mean_diff = float(diff.mean())
        p_exact = exact_signed_rank_p(diff)
        r_rb = rank_biserial(diff)
        hl = hodges_lehmann(diff)
        ci_low, ci_high = bootstrap_hl_ci(diff, n_bootstrap=args.bootstrap, seed=args.seed)

        # Scipy cross-check
        res_scipy = stats.wilcoxon(left_run, right_run, alternative="two-sided")
        p_scipy = float(res_scipy.pvalue)
        w_stat = float(res_scipy.statistic)

        stat_rows_full.append({
            "Metric": metric,
            "Comparator": f"{left_name} vs {right_name}",
            "Left": left_name,
            "Right": right_name,
            "Mean diff": mean_diff,
            "Wilcoxon W": w_stat,
            "Wilcoxon p": p_exact,
            "Wilcoxon p (scipy)": p_scipy,
            "Effect size r_rb": r_rb,
            "HL diff": hl,
            "HL 95% CI low": ci_low,
            "HL 95% CI high": ci_high,
            "HL 95% CI": f"[{ci_low:.6f}, {ci_high:.6f}]",
        })

    # BH correction across all 4 comparisons
    p_vals = [r["Wilcoxon p"] for r in stat_rows_full]
    q_vals = bh_adjust(p_vals)
    for r, q in zip(stat_rows_full, q_vals):
        r["BH q"] = float(q)
        # Automated conclusion
        if q < 0.05 and r["HL diff"] > 0:
            conclusion = "Left significantly higher"
        elif q < 0.05 and r["HL diff"] < 0:
            conclusion = "Right significantly higher"
        else:
            conclusion = "No statistically significant difference"
        r["Conclusion"] = conclusion

    df_full = pd.DataFrame(stat_rows_full)

    # 4. Formatted Table R5 matching required output columns
    formatted_rows = []
    for r in stat_rows_full:
        formatted_rows.append({
            "Metric": r["Metric"],
            "Comparator": r["Comparator"],
            "Mean diff": f"{r['Mean diff']:+.4f}",
            "Wilcoxon p": f"{r['Wilcoxon p']:.3E}",
            "BH q": f"{r['BH q']:.3E}",
            "Effect size r_rb": f"{r['Effect size r_rb']:+.3f}",
            "HL diff": f"{r['HL diff']:+.6f}",
            "HL 95% CI": r["HL 95% CI"],
            "Conclusion": r["Conclusion"],
        })
    df_formatted = pd.DataFrame(formatted_rows)

    # Save CSVs
    csv_formatted_path = args.output_dir / "table_R5_label_masking_formatted.csv"
    csv_full_path = args.output_dir / "table_R5_label_masking_full_precision.csv"
    csv_desc_path = args.output_dir / "table_R5_descriptive_summary.csv"
    csv_runs_path = args.output_dir / "table_R5_run_level_means.csv"

    df_formatted.to_csv(csv_formatted_path, index=False)
    df_full.to_csv(csv_full_path, index=False)
    df_desc.to_csv(csv_desc_path, index=False)
    df_runs.to_csv(csv_runs_path, index=False)

    # Save Excel Workbook with professional styling
    excel_path = args.output_dir / "Table_R5_Statistical_Validation.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_formatted.to_excel(writer, sheet_name="Table_R5_Formatted", index=False)
        df_full.to_excel(writer, sheet_name="Table_R5_Full_Precision", index=False)
        df_desc.to_excel(writer, sheet_name="Descriptive_Statistics", index=False)
        df_runs.to_excel(writer, sheet_name="Run_Level_10_Means", index=False)

        # Raw fold-level matrices sheet
        raw_rows = []
        for run in range(10):
            for fold in range(5):
                row = {"Run": run + 1, "Fold": fold + 1}
                for model_name in ["Original Transductive BIMODriver", "Masked Transductive", "Masked Inductive"]:
                    for metric in ["AUROC", "AUPRC"]:
                        col_name = f"{model_name} ({metric})"
                        row[col_name] = float(data[model_name][metric][run, fold])
                raw_rows.append(row)
        df_raw = pd.DataFrame(raw_rows)
        df_raw.to_excel(writer, sheet_name="Raw_50fold_Data", index=False)

        # Format sheets
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F497D")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for col in sheet.columns:
                values = [str(c.value) if c.value is not None else "" for c in col]
                width = min(max(max(map(len, values)) + 3, 12), 50)
                sheet.column_dimensions[col[0].column_letter].width = width

    # Save JSON archive
    json_path = args.output_dir / "table_R5_summary.json"
    summary_payload = {
        "metadata": {
            "title": "Table R5: Label-Like Phrases Masking Statistical Validation",
            "dataset": "CPDB Pan-Cancer (13,627 genes)",
            "n_runs": 10,
            "n_folds_per_run": 5,
            "total_folds": 50,
            "unit_of_analysis": "10 paired run-level means",
            "bootstrap_samples": args.bootstrap,
            "seed": args.seed,
            "sd_ddof": 0,
            "input_file_hashes_sha256": hashes,
            "protocol": {
                "test": "Two-sided paired Wilcoxon signed-rank test",
                "multiple_testing": "Benjamini-Hochberg FDR correction across m=4 tests",
                "effect_size": "Paired rank-biserial correlation (r_rb)",
                "location_shift": "Hodges-Lehmann difference (Walsh averages median)",
                "ci": "95% bootstrap CI (10,000 resamples)",
            },
        },
        "descriptive_summary": df_desc.to_dict(orient="records"),
        "table_R5_full_precision": df_full.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSuccessfully generated all Table R5 statistical deliverables:")
    print(f"  - Formatted CSV: {csv_formatted_path}")
    print(f"  - Full Precision CSV: {csv_full_path}")
    print(f"  - Descriptive Summary CSV: {csv_desc_path}")
    print(f"  - Run-Level Means CSV: {csv_runs_path}")
    print(f"  - Excel Workbook: {excel_path}")
    print(f"  - JSON Report: {json_path}")
    print("\nTable R5 Formatted Preview:")
    print(df_formatted.to_string(index=False))


if __name__ == "__main__":
    main()
