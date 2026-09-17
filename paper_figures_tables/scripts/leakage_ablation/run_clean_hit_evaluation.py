#!/usr/bin/env python3
"""Statistical significance testing for Clean <-> Hit baseline benchmark experiments.

Supports both One-sided (BIMODriver > Baseline) and Two-sided paired Wilcoxon signed-rank tests.
Compares BIMODriver against MNGCL and DISFusion on cross-distribution tasks:
- clean_to_hit (Train 2,402 clean genes -> Test 581 hit genes)
- hit_to_clean (Train 581 hit genes -> Test 2,402 clean genes)
under both:
- Transductive (full graph propagation during training)
- Strict Inductive (edges cut, test features zeroed during training)

Computes:
- Descriptive stats (Mean, SD ddof=0, SD ddof=1, Median, Min, Max)
- Exact paired one-sided (greater) and two-sided Wilcoxon signed-rank tests
- Paired Rank-Biserial correlation r_rb
- Paired Hodges-Lehmann difference with 95% bootstrap CI (10,000 resamples)
- Benjamini-Hochberg (BH) FDR corrections for both one-sided and two-sided families
Outputs structured CSVs, publication-ready Excel workbook, and JSON archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from scipy import stats


def exact_signed_rank_p(diff: np.ndarray, alternative: str = "greater") -> float:
    """Exact sign-flip permutation p-value handling ties.

    alternative: 'greater' (H1: diff > 0, i.e. BIMODriver > Baseline) or 'two-sided'
    """
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
    if alternative == "greater":
        return float(np.mean(null >= observed - 1e-12))
    elif alternative == "two-sided":
        center = float(ranks.sum()) / 2.0
        return float(np.mean(np.abs(null - center) >= abs(observed - center) - 1e-12))
    elif alternative == "less":
        return float(np.mean(null <= observed + 1e-12))
    else:
        raise ValueError(f"Unknown alternative: {alternative}")


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
    seed: int = 20260912,
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


def deterministic_seed(key: str, base_seed: int) -> int:
    token = hashlib.sha256(f"{base_seed}|{key}".encode()).hexdigest()[:8]
    return int(token, 16)


def format_significance_stars(p_val: float, q_val: float) -> str:
    """Return asterisks if both p and q are significant."""
    if q_val >= 0.05:
        return ""
    if p_val < 0.005:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return ""


def run_evaluation(
    data_dir: Path,
    output_dir: Path,
    n_bootstrap: int = 10_000,
    base_seed: int = 20260912,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("clean_to_hit", "Clean → Hit", "Clean (2,402) -> Hit (581)"),
        ("hit_to_clean", "Hit → Clean", "Hit (581) -> Clean (2,402)"),
    ]
    paradigms = [
        ("Transductive", "leakage", "传导式"),
        ("Strict Inductive", "inductive_leakage", "严格归纳式"),
    ]
    metrics = [
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
    ]
    models = [
        ("BIMODriver", "bimodriver_original", "本文模型"),
        ("MNGCL", "mngcl", "对比基准"),
        ("DISFusion", "disfusion", "对比基准"),
    ]

    # 1. Load data arrays
    arrays: dict[tuple[str, str, str, str], np.ndarray] = {}
    for t_key, _, _ in tasks:
        for p_name, p_stem, _ in paradigms:
            for m_name, m_stem in metrics:
                for mod_name, mod_stem, _ in models:
                    fn = f"{mod_stem}_{p_stem}_{t_key}_{m_stem}.txt"
                    fp = data_dir / fn
                    if not fp.exists():
                        raise FileNotFoundError(f"Missing required file: {fp}")
                    arr = np.asarray(np.loadtxt(fp, dtype=float), dtype=float)
                    if arr.shape != (10,):
                        raise ValueError(f"{fp}: expected shape (10,), got {arr.shape}")
                    arrays[(mod_name, t_key, p_name, m_name)] = arr

    # 2. Descriptive statistics table
    desc_rows = []
    for t_key, t_display, t_desc in tasks:
        for p_name, _, p_desc in paradigms:
            for m_name, _ in metrics:
                for mod_name, _, mod_role in models:
                    arr = arrays[(mod_name, t_key, p_name, m_name)]
                    desc_rows.append({
                        "Task": t_key,
                        "Task Display": t_display,
                        "Task Description": t_desc,
                        "Paradigm": p_name,
                        "Paradigm (CN)": p_desc,
                        "Model": mod_name,
                        "Model Role": mod_role,
                        "Metric": m_name,
                        "N": len(arr),
                        "Mean": float(arr.mean()),
                        "SD (ddof=0)": float(arr.std(ddof=0)),
                        "SD (ddof=1)": float(arr.std(ddof=1)),
                        "Median": float(np.median(arr)),
                        "Min": float(arr.min()),
                        "Max": float(arr.max()),
                        "Mean ± SD": f"{arr.mean():.4f} ± {arr.std(ddof=0):.4f}",
                    })
    df_desc = pd.DataFrame(desc_rows)

    # 3. Pairwise Statistical Comparisons (Both One-sided and Two-sided)
    comp_rows = []
    for t_key, t_display, _ in tasks:
        for p_name, _, _ in paradigms:
            for m_name, _ in metrics:
                bm_arr = arrays[("BIMODriver", t_key, p_name, m_name)]
                for base_name in ["MNGCL", "DISFusion"]:
                    base_arr = arrays[(base_name, t_key, p_name, m_name)]
                    diff = bm_arr - base_arr
                    p_one = exact_signed_rank_p(diff, alternative="greater")
                    p_two = exact_signed_rank_p(diff, alternative="two-sided")
                    r_rb = rank_biserial(diff)
                    hl = hodges_lehmann(diff)

                    seed_key = f"{t_key}|{p_name}|{m_name}|BIMODriver_vs_{base_name}"
                    ci_low, ci_high = bootstrap_hl_ci(
                        diff,
                        n_bootstrap=n_bootstrap,
                        seed=deterministic_seed(seed_key, base_seed),
                    )

                    comp_rows.append({
                        "Task": t_key,
                        "Task Display": t_display,
                        "Paradigm": p_name,
                        "Metric": m_name,
                        "Comparison": f"BIMODriver - {base_name}",
                        "Candidate": "BIMODriver",
                        "Baseline": base_name,
                        "N": len(diff),
                        "BIMODriver Mean": float(bm_arr.mean()),
                        "BIMODriver SD (ddof=0)": float(bm_arr.std(ddof=0)),
                        "Baseline Mean": float(base_arr.mean()),
                        "Baseline SD (ddof=0)": float(base_arr.std(ddof=0)),
                        "Mean Difference": float(diff.mean()),
                        "Hodges-Lehmann Diff": hl,
                        "95% CI Low": ci_low,
                        "95% CI High": ci_high,
                        "Rank-Biserial r_rb": r_rb,
                        "Wilcoxon p (one-sided, greater)": p_one,
                        "Wilcoxon p (two-sided)": p_two,
                    })

    # BH FDR corrections for One-Sided:
    q_one_glob = bh_adjust([r["Wilcoxon p (one-sided, greater)"] for r in comp_rows])
    for r, q in zip(comp_rows, q_one_glob):
        r["BH q (one-sided, Global, m=16)"] = float(q)
        r["Significant One-Sided Global (q<0.05)"] = bool(q < 0.05)

    for p_name in ["Transductive", "Strict Inductive"]:
        indices = [i for i, r in enumerate(comp_rows) if r["Paradigm"] == p_name]
        q_p_one = bh_adjust([comp_rows[i]["Wilcoxon p (one-sided, greater)"] for i in indices])
        for idx, q in zip(indices, q_p_one):
            comp_rows[idx]["BH q (one-sided, By-Paradigm, m=8)"] = float(q)
            comp_rows[idx]["Significant One-Sided By-Paradigm (q<0.05)"] = bool(q < 0.05)

    # BH FDR corrections for Two-Sided:
    q_two_glob = bh_adjust([r["Wilcoxon p (two-sided)"] for r in comp_rows])
    for r, q in zip(comp_rows, q_two_glob):
        r["BH q (two-sided, Global, m=16)"] = float(q)
        r["Significant Two-Sided Global (q<0.05)"] = bool(q < 0.05)

    for p_name in ["Transductive", "Strict Inductive"]:
        indices = [i for i, r in enumerate(comp_rows) if r["Paradigm"] == p_name]
        q_p_two = bh_adjust([comp_rows[i]["Wilcoxon p (two-sided)"] for i in indices])
        for idx, q in zip(indices, q_p_two):
            comp_rows[idx]["BH q (two-sided, By-Paradigm, m=8)"] = float(q)
            comp_rows[idx]["Significant Two-Sided By-Paradigm (q<0.05)"] = bool(q < 0.05)

    df_comp = pd.DataFrame(comp_rows)

    # Separate into dedicated Transductive and Strict Inductive tables
    df_trans = df_comp[df_comp["Paradigm"] == "Transductive"].copy()
    df_ind = df_comp[df_comp["Paradigm"] == "Strict Inductive"].copy()

    # 4. Publication-ready Main Paper Table (One-sided - Primary for paper body)
    main_paper_onesided = []
    main_paper_twosided = []

    for t_key, t_display, _ in tasks:
        for p_name, _, _ in paradigms:
            for m_name, _ in metrics:
                bm_arr = arrays[("BIMODriver", t_key, p_name, m_name)]
                mng_arr = arrays[("MNGCL", t_key, p_name, m_name)]
                dis_arr = arrays[("DISFusion", t_key, p_name, m_name)]

                # Lookup MNGCL stats
                row_mng = df_comp[
                    (df_comp["Task"] == t_key)
                    & (df_comp["Paradigm"] == p_name)
                    & (df_comp["Metric"] == m_name)
                    & (df_comp["Baseline"] == "MNGCL")
                ].iloc[0]

                # Lookup DISFusion stats
                row_dis = df_comp[
                    (df_comp["Task"] == t_key)
                    & (df_comp["Paradigm"] == p_name)
                    & (df_comp["Metric"] == m_name)
                    & (df_comp["Baseline"] == "DISFusion")
                ].iloc[0]

                diff_mng = bm_arr.mean() - mng_arr.mean()
                diff_dis = bm_arr.mean() - dis_arr.mean()

                # Stars based on one-sided
                star_mng_one = format_significance_stars(
                    row_mng["Wilcoxon p (one-sided, greater)"],
                    row_mng["BH q (one-sided, By-Paradigm, m=8)"],
                )
                star_dis_one = format_significance_stars(
                    row_dis["Wilcoxon p (one-sided, greater)"],
                    row_dis["BH q (one-sided, By-Paradigm, m=8)"],
                )

                # Stars based on two-sided
                star_mng_two = format_significance_stars(
                    row_mng["Wilcoxon p (two-sided)"],
                    row_mng["BH q (two-sided, By-Paradigm, m=8)"],
                )
                star_dis_two = format_significance_stars(
                    row_dis["Wilcoxon p (two-sided)"],
                    row_dis["BH q (two-sided, By-Paradigm, m=8)"],
                )

                def fmt_stat(val: float) -> str:
                    if pd.isna(val):
                        return "N/A"
                    if val < 1e-15:
                        return "< 1.00E-15"
                    if val < 0.001:
                        return f"{val:.2E}"
                    return f"{val:.4f}"

                main_paper_onesided.append({
                    "Evaluation Task": t_display,
                    "Graph Learning Setting": p_name,
                    "Metric": m_name,
                    "BIMODriver (Mean ± SD)": f"{bm_arr.mean():.4f} ± {bm_arr.std(ddof=0):.4f}",
                    "MNGCL (Mean ± SD)": f"{mng_arr.mean():.4f} ± {mng_arr.std(ddof=0):.4f}",
                    "Diff vs MNGCL": f"{diff_mng:+.4f}{(' ' + star_mng_one) if star_mng_one else ''}",
                    "Wilcoxon p (vs MNGCL, one-sided)": fmt_stat(float(row_mng["Wilcoxon p (one-sided, greater)"])),
                    "BH q (vs MNGCL, one-sided)": fmt_stat(float(row_mng["BH q (one-sided, By-Paradigm, m=8)"])),
                    "DISFusion (Mean ± SD)": f"{dis_arr.mean():.4f} ± {dis_arr.std(ddof=0):.4f}",
                    "Diff vs DISFusion": f"{diff_dis:+.4f}{(' ' + star_dis_one) if star_dis_one else ''}",
                    "Wilcoxon p (vs DISFusion, one-sided)": fmt_stat(float(row_dis["Wilcoxon p (one-sided, greater)"])),
                    "BH q (vs DISFusion, one-sided)": fmt_stat(float(row_dis["BH q (one-sided, By-Paradigm, m=8)"])),
                })

                main_paper_twosided.append({
                    "Evaluation Task": t_display,
                    "Graph Learning Setting": p_name,
                    "Metric": m_name,
                    "BIMODriver (Mean ± SD)": f"{bm_arr.mean():.4f} ± {bm_arr.std(ddof=0):.4f}",
                    "MNGCL (Mean ± SD)": f"{mng_arr.mean():.4f} ± {mng_arr.std(ddof=0):.4f}",
                    "Diff vs MNGCL": f"{diff_mng:+.4f}{(' ' + star_mng_two) if star_mng_two else ''}",
                    "Wilcoxon p (vs MNGCL, two-sided)": fmt_stat(float(row_mng["Wilcoxon p (two-sided)"])),
                    "BH q (vs MNGCL, two-sided)": fmt_stat(float(row_mng["BH q (two-sided, By-Paradigm, m=8)"])),
                    "DISFusion (Mean ± SD)": f"{dis_arr.mean():.4f} ± {dis_arr.std(ddof=0):.4f}",
                    "Diff vs DISFusion": f"{diff_dis:+.4f}{(' ' + star_dis_two) if star_dis_two else ''}",
                    "Wilcoxon p (vs DISFusion, two-sided)": fmt_stat(float(row_dis["Wilcoxon p (two-sided)"])),
                    "BH q (vs DISFusion, two-sided)": fmt_stat(float(row_dis["BH q (two-sided, By-Paradigm, m=8)"])),
                })

    df_main_paper_onesided = pd.DataFrame(main_paper_onesided)
    df_main_paper_twosided = pd.DataFrame(main_paper_twosided)

    # 5. Raw Run-by-run scores
    raw_rows = []
    for run_idx in range(10):
        for t_key, t_display, _ in tasks:
            for p_name, _, _ in paradigms:
                for m_name, _ in metrics:
                    bm_val = float(arrays[("BIMODriver", t_key, p_name, m_name)][run_idx])
                    mng_val = float(arrays[("MNGCL", t_key, p_name, m_name)][run_idx])
                    dis_val = float(arrays[("DISFusion", t_key, p_name, m_name)][run_idx])
                    raw_rows.append({
                        "Run": run_idx + 1,
                        "Task": t_key,
                        "Task Display": t_display,
                        "Paradigm": p_name,
                        "Metric": m_name,
                        "BIMODriver": bm_val,
                        "MNGCL": mng_val,
                        "DISFusion": dis_val,
                        "Diff (BM - MNGCL)": bm_val - mng_val,
                        "Diff (BM - DISFusion)": bm_val - dis_val,
                    })
    df_raw = pd.DataFrame(raw_rows)

    # Save CSVs (Two-sided is primary paper main table per unified standard)
    df_main_paper_twosided.to_csv(output_dir / "clean_hit_paper_main_table.csv", index=False)
    df_main_paper_twosided.to_csv(output_dir / "clean_hit_paper_main_table_two_sided.csv", index=False)
    df_main_paper_onesided.to_csv(output_dir / "clean_hit_paper_supplementary_one_sided.csv", index=False)
    df_comp.to_csv(output_dir / "clean_hit_statistical_comparison.csv", index=False)
    df_trans.to_csv(output_dir / "clean_hit_transductive_comparison.csv", index=False)
    df_ind.to_csv(output_dir / "clean_hit_inductive_comparison.csv", index=False)
    df_desc.to_csv(output_dir / "clean_hit_descriptive_summary.csv", index=False)
    df_raw.to_csv(output_dir / "clean_hit_raw_scores.csv", index=False)

    # Save Excel Workbook with professional styling (Main_Table_TwoSided as Sheet 1)
    excel_path = output_dir / "Clean_Hit_Statistical_Validation.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_main_paper_twosided.to_excel(writer, sheet_name="Main_Table_TwoSided", index=False)
        df_main_paper_onesided.to_excel(writer, sheet_name="Supplementary_Table_OneSided", index=False)
        df_trans.to_excel(writer, sheet_name="Transductive_Comparisons", index=False)
        df_ind.to_excel(writer, sheet_name="Inductive_Comparisons", index=False)
        df_comp.to_excel(writer, sheet_name="All_Comparisons_Detailed", index=False)
        df_desc.to_excel(writer, sheet_name="Descriptive_Summary", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_Run_Scores", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F497D")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for column_cells in sheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max(map(len, values)) + 3, 12), 40)
                sheet.column_dimensions[column_cells[0].column_letter].width = width

    # Save JSON Archive
    json_path = output_dir / "clean_hit_statistical_summary.json"
    summary_payload = {
        "metadata": {
            "title": "Clean <-> Hit Benchmark Statistical Significance Testing (One-Sided Primary & Two-Sided Supplementary)",
            "tasks": ["clean_to_hit", "hit_to_clean"],
            "paradigms": ["Transductive", "Strict Inductive"],
            "models": ["BIMODriver", "MNGCL", "DISFusion"],
            "metrics": ["AUROC", "AUPRC"],
            "n_runs": 10,
            "bootstrap_samples": n_bootstrap,
            "base_seed": base_seed,
            "methodology": {
                "primary_test": "Exact one-sided paired Wilcoxon signed-rank test (H1: BIMODriver > Baseline)",
                "supplementary_test": "Exact two-sided paired Wilcoxon signed-rank test",
                "effect_size": "Paired rank-biserial correlation r_rb",
                "difference_estimate": "Hodges-Lehmann difference (median of Walsh averages)",
                "confidence_interval": "95% percentile bootstrap interval",
                "multiple_testing": "Benjamini-Hochberg FDR (m=8 per-paradigm and m=16 global)",
            },
        },
        "pairwise_comparisons": df_comp.to_dict(orient="records"),
        "main_paper_table_onesided": df_main_paper_onesided.to_dict(orient="records"),
        "main_paper_table_twosided": df_main_paper_twosided.to_dict(orient="records"),
        "descriptive_summary": df_desc.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Successfully generated all deliverables in: {output_dir}")
    print(f"  - Main Paper Table (One-Sided): {output_dir / 'clean_hit_paper_main_table.csv'}")
    print(f"  - Main Paper Table (Two-Sided): {output_dir / 'clean_hit_paper_main_table_two_sided.csv'}")
    print(f"  - Statistical Comparisons: {output_dir / 'clean_hit_statistical_comparison.csv'}")
    print(f"  - Transductive Comparisons: {output_dir / 'clean_hit_transductive_comparison.csv'}")
    print(f"  - Inductive Comparisons: {output_dir / 'clean_hit_inductive_comparison.csv'}")
    print(f"  - Descriptive Summary: {output_dir / 'clean_hit_descriptive_summary.csv'}")
    print(f"  - Raw Run Scores: {output_dir / 'clean_hit_raw_scores.csv'}")
    print(f"  - Excel Workbook: {excel_path}")
    print(f"  - JSON Report: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean <-> Hit statistical significance testing")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "results" / "clean_hit_results_final",
        help="Directory containing the 10-run result files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "results" / "clean_hit_results_final",
        help="Output directory for generated tables and workbook",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
