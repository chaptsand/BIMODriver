#!/usr/bin/env python3
"""Statistical validation for Table A19: Edge-Masking Analysis across Transductive & Masking Variants.

Strictly aligned with latest paper appendix (appendix.tex / Table A19).
Computes Scheme C:
1. Primary Table A19 performance comparison (BIMODriver-Original, BIMODriver-Masking, DISFusion, MNGCL).
2. Extended Table A19 summary statistics (50-fold and 10-run across 7 variants).
3. Masking Variants vs Baselines (DISFusion and MNGCL) under both 10-run (exact) and 50-fold (asymptotic) Wilcoxon tests.
4. Masking Ablation vs Original Reference (BIMODriver-Original) under both 10-run (exact) and 50-fold (asymptotic) Wilcoxon tests.
Generates comprehensive CSVs, formatted Excel workbook, and JSON archive.
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


def asymptotic_signed_rank_p(diff: np.ndarray) -> float:
    """Asymptotic two-sided Wilcoxon signed-rank test for N=50."""
    d = np.asarray(diff, dtype=float)
    if np.all(np.isclose(d, 0.0, atol=1e-15)):
        return 1.0
    return float(
        stats.wilcoxon(
            d,
            alternative="two-sided",
            zero_method="wilcox",
            correction=False,
            method="asymptotic",
        ).pvalue
    )


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


def load_all_data(data_root: Path) -> dict[tuple[str, str, str], np.ndarray]:
    specs = [
        ("BIMODriver-Original", "CPDB", data_root / "BIMODriver-Original" / "BIMODriver-Original-CPDB.txt", False),
        ("BIMODriver-Original", "STRING", data_root / "BIMODriver-Original" / "BIMODriver-Original-STRING.txt", False),
        ("BIMODriver-Masking", "CPDB", data_root / "BIMODriver-Masking" / "严格归纳式" / "BIMODriver-Masking-CPDB.txt", False),
        ("BIMODriver-Masking", "STRING", data_root / "BIMODriver-Masking" / "严格归纳式" / "BIMODriver-Masking-STRING.txt", False),
        ("BIMODriver-Masking (Mask Test-Incident Edges)", "CPDB", data_root / "BIMODriver-Masking" / "去掉测试集相连的边" / "cpdb" / "pan-cancer.txt", False),
        ("BIMODriver-Masking (Mask Test-Incident Edges)", "STRING", data_root / "BIMODriver-Masking" / "去掉测试集相连的边" / "string" / "pan-cancer_string.txt", False),
        ("BIMODriver-Masking (Exclude Test Nodes from Loss)", "CPDB", data_root / "BIMODriver-Masking" / "测试节点不参加对比损失" / "cpdb" / "pan-cancer.txt", False),
        ("BIMODriver-Masking (Exclude Test Nodes from Loss)", "STRING", data_root / "BIMODriver-Masking" / "测试节点不参加对比损失" / "string" / "pan-cancer_string.txt", False),
        ("BIMODriver-Masking (Mask All Graph Edges)", "CPDB", data_root / "BIMODriver-Masking" / "去掉所有的边" / "cpdb" / "pan-cancer.txt", False),
        ("BIMODriver-Masking (Mask All Graph Edges)", "STRING", data_root / "BIMODriver-Masking" / "去掉所有的边" / "string" / "pan-cancer_string.txt", False),
        ("DISFusion", "CPDB", data_root / "DISFusion" / "DISFusion-CPDB.txt", True),
        ("DISFusion", "STRING", data_root / "DISFusion" / "DISFusion-STRING.txt", True),
        ("MNGCL", "CPDB", data_root / "MNGCL" / "MNGCL-CPDB.txt", True),
        ("MNGCL", "STRING", data_root / "MNGCL" / "MNGCL-STRING.txt", True),
    ]

    data = {}
    for model, dataset, path, auc_first in specs:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines:
            cleaned = line.replace("[", " ").replace("]", " ")
            parts = cleaned.split()
            r = []
            for x in parts:
                try:
                    v = float(x)
                    if 0.0 <= v <= 1.0:
                        r.append(v)
                except ValueError:
                    pass
            if len(r) == 5:
                rows.append(r)
        if len(rows) != 20:
            raise ValueError(f"{path}: expected 20 rows of 5 values, got {len(rows)}")

        if auc_first:
            auc_mat = np.array(rows[:10], dtype=float)
            auprc_mat = np.array(rows[10:20], dtype=float)
        else:
            auprc_mat = np.array(rows[:10], dtype=float)
            auc_mat = np.array(rows[10:20], dtype=float)

        data[(model, dataset, "AUPRC")] = auprc_mat
        data[(model, dataset, "AUC")] = auc_mat

    return data


def compute_comparison_row(
    left_arr: np.ndarray,
    right_arr: np.ndarray,
    left_name: str,
    right_name: str,
    dataset: str,
    metric: str,
    paired_unit: str,
    exact: bool,
    n_bootstrap: int,
    base_seed: int,
) -> dict[str, object]:
    diff = left_arr - right_arr
    p_val = exact_signed_rank_p(diff) if exact else asymptotic_signed_rank_p(diff)
    r_rb = rank_biserial(diff)
    hl = hodges_lehmann(diff)

    seed_key = f"{dataset}|{metric}|{left_name}_vs_{right_name}|{paired_unit}"
    ci_low, ci_high = bootstrap_hl_ci(
        diff,
        n_bootstrap=n_bootstrap,
        seed=deterministic_seed(seed_key, base_seed),
    )

    return {
        "Dataset": dataset,
        "Metric": metric,
        "Comparison": f"{left_name} - {right_name}",
        "Candidate": left_name,
        "Reference": right_name,
        "Paired Unit": paired_unit,
        "N": len(diff),
        "Candidate Mean": float(left_arr.mean()),
        "Candidate SD (ddof=0)": float(left_arr.std(ddof=0)),
        "Reference Mean": float(right_arr.mean()),
        "Reference SD (ddof=0)": float(right_arr.std(ddof=0)),
        "Mean Difference": float(diff.mean()),
        "Hodges-Lehmann Diff": hl,
        "95% CI Low": ci_low,
        "95% CI High": ci_high,
        "Rank-Biserial r_rb": r_rb,
        "Wilcoxon p (two-sided)": p_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheme C statistical testing for Table A19")
    repo_root = Path(__file__).resolve().parents[2]
    default_data = repo_root / "data" / "edge_masking_A19"

    parser.add_argument("--data-root", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=repo_root / "results" / "edge_masking_A19")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_all_data(args.data_root)

    models = [
        "BIMODriver-Original",
        "BIMODriver-Masking",
        "BIMODriver-Masking (Mask Test-Incident Edges)",
        "BIMODriver-Masking (Exclude Test Nodes from Loss)",
        "BIMODriver-Masking (Mask All Graph Edges)",
        "DISFusion",
        "MNGCL",
    ]
    bimodriver_variants = models[:5]
    baselines = ["DISFusion", "MNGCL"]
    masking_variants = models[1:5]
    datasets = ["CPDB", "STRING"]
    metrics = ["AUPRC", "AUC"]

    # 1. Primary Paper Table A19 (4 models matching published Table A19)
    paper_models = ["BIMODriver-Original", "BIMODriver-Masking", "DISFusion", "MNGCL"]
    paper_a19_rows = []
    for m in paper_models:
        c_auprc = data[(m, "CPDB", "AUPRC")]
        c_auc = data[(m, "CPDB", "AUC")]
        s_auprc = data[(m, "STRING", "AUPRC")]
        s_auc = data[(m, "STRING", "AUC")]
        paper_a19_rows.append({
            "Model Variant": m,
            "CPDB AUPRC": f"{c_auprc.mean():.4f} ± {c_auprc.std(ddof=0):.4f}",
            "CPDB AUC": f"{c_auc.mean():.4f} ± {c_auc.std(ddof=0):.4f}",
            "STRING AUPRC": f"{s_auprc.mean():.4f} ± {s_auprc.std(ddof=0):.4f}",
            "STRING AUC": f"{s_auc.mean():.4f} ± {s_auc.std(ddof=0):.4f}",
        })
    df_paper_a19 = pd.DataFrame(paper_a19_rows)

    # 2. Extended Table A19 (all 7 variants)
    extended_a19_rows = []
    for m in models:
        c_auprc = data[(m, "CPDB", "AUPRC")]
        c_auc = data[(m, "CPDB", "AUC")]
        s_auprc = data[(m, "STRING", "AUPRC")]
        s_auc = data[(m, "STRING", "AUC")]
        extended_a19_rows.append({
            "Model Variant": m,
            "CPDB AUPRC": f"{c_auprc.mean():.4f} ± {c_auprc.std(ddof=0):.4f}",
            "CPDB AUC": f"{c_auc.mean():.4f} ± {c_auc.std(ddof=0):.4f}",
            "STRING AUPRC": f"{s_auprc.mean():.4f} ± {s_auprc.std(ddof=0):.4f}",
            "STRING AUC": f"{s_auc.mean():.4f} ± {s_auc.std(ddof=0):.4f}",
        })
    df_extended_a19 = pd.DataFrame(extended_a19_rows)

    # Descriptive Summary Table
    summary_rows = []
    for m in models:
        for d in datasets:
            for met in metrics:
                mat = data[(m, d, met)]
                f50 = mat.reshape(-1)
                r10 = mat.mean(axis=1)
                summary_rows.append({
                    "Model Variant": m,
                    "Dataset": d,
                    "Metric": met,
                    "50-Fold Mean": float(f50.mean()),
                    "50-Fold SD": float(f50.std(ddof=0)),
                    "10-Run Mean": float(r10.mean()),
                    "10-Run SD": float(r10.std(ddof=0)),
                    "Min Fold": float(f50.min()),
                    "Max Fold": float(f50.max()),
                })
    df_summary = pd.DataFrame(summary_rows)

    # 3. Part 1: Masking Variants vs Baselines (10-run and 50-fold)
    vs_base_10run = []
    vs_base_50fold = []

    for d in datasets:
        for met in metrics:
            for mask in bimodriver_variants:
                mask_mat = data[(mask, d, met)]
                for base in baselines:
                    base_mat = data[(base, d, met)]

                    # 10-run mean (Variant minus Baseline)
                    vs_base_10run.append(compute_comparison_row(
                        left_arr=mask_mat.mean(axis=1),
                        right_arr=base_mat.mean(axis=1),
                        left_name=mask,
                        right_name=base,
                        dataset=d,
                        metric=met,
                        paired_unit="10-run mean",
                        exact=True,
                        n_bootstrap=args.bootstrap,
                        base_seed=args.seed,
                    ))

                    # 50-fold value (Variant minus Baseline)
                    vs_base_50fold.append(compute_comparison_row(
                        left_arr=mask_mat.reshape(-1),
                        right_arr=base_mat.reshape(-1),
                        left_name=mask,
                        right_name=base,
                        dataset=d,
                        metric=met,
                        paired_unit="50-fold value",
                        exact=False,
                        n_bootstrap=args.bootstrap,
                        base_seed=args.seed,
                    ))

    # Apply BH FDR corrections
    for row_list in (vs_base_10run, vs_base_50fold):
        q_glob = bh_adjust([r["Wilcoxon p (two-sided)"] for r in row_list])
        for r, q in zip(row_list, q_glob):
            r["BH q (Global, m=40)"] = float(q)
            r["Significant Global (q<0.05)"] = bool(q < 0.05)

        for d in datasets:
            for met in metrics:
                idxs = [i for i, r in enumerate(row_list) if r["Dataset"] == d and r["Metric"] == met]
                sub_p = [row_list[i]["Wilcoxon p (two-sided)"] for i in idxs]
                q_sub = bh_adjust(sub_p)
                for idx, q in zip(idxs, q_sub):
                    row_list[idx]["BH q (Family, m=10)"] = float(q)
                    row_list[idx]["Significant Family (q<0.05)"] = bool(q < 0.05)

    df_vs_base_10run = pd.DataFrame(vs_base_10run)
    df_vs_base_50fold = pd.DataFrame(vs_base_50fold)

    # 4. Part 2: Masking Ablation vs Original Reference (10-run and 50-fold)
    ablation_10run = []
    ablation_50fold = []
    orig_name = "BIMODriver-Original"

    for d in datasets:
        for met in metrics:
            orig_mat = data[(orig_name, d, met)]
            for mask in masking_variants:
                mask_mat = data[(mask, d, met)]

                # 10-run mean (Variant minus Original)
                ablation_10run.append(compute_comparison_row(
                    left_arr=mask_mat.mean(axis=1),
                    right_arr=orig_mat.mean(axis=1),
                    left_name=mask,
                    right_name=orig_name,
                    dataset=d,
                    metric=met,
                    paired_unit="10-run mean",
                    exact=True,
                    n_bootstrap=args.bootstrap,
                    base_seed=args.seed,
                ))

                # 50-fold value (Variant minus Original)
                ablation_50fold.append(compute_comparison_row(
                    left_arr=mask_mat.reshape(-1),
                    right_arr=orig_mat.reshape(-1),
                    left_name=mask,
                    right_name=orig_name,
                    dataset=d,
                    metric=met,
                    paired_unit="50-fold value",
                    exact=False,
                    n_bootstrap=args.bootstrap,
                    base_seed=args.seed,
                ))

    # Apply BH FDR corrections
    for row_list in (ablation_10run, ablation_50fold):
        q_glob = bh_adjust([r["Wilcoxon p (two-sided)"] for r in row_list])
        for r, q in zip(row_list, q_glob):
            r["BH q (Global, m=16)"] = float(q)
            r["Significant Global (q<0.05)"] = bool(q < 0.05)

        for d in datasets:
            for met in metrics:
                idxs = [i for i, r in enumerate(row_list) if r["Dataset"] == d and r["Metric"] == met]
                sub_p = [row_list[i]["Wilcoxon p (two-sided)"] for i in idxs]
                q_sub = bh_adjust(sub_p)
                for idx, q in zip(idxs, q_sub):
                    row_list[idx]["BH q (Family, m=4)"] = float(q)
                    row_list[idx]["Significant Family (q<0.05)"] = bool(q < 0.05)

    df_ablation_10run = pd.DataFrame(ablation_10run)
    df_ablation_50fold = pd.DataFrame(ablation_50fold)

    # 5. Raw matrices
    raw_rows = []
    for run in range(10):
        for fold in range(5):
            for d in datasets:
                for met in metrics:
                    row = {
                        "Run": run + 1,
                        "Fold": fold + 1,
                        "Dataset": d,
                        "Metric": met,
                    }
                    for m in models:
                        row[m] = float(data[(m, d, met)][run, fold])
                    raw_rows.append(row)
    df_raw = pd.DataFrame(raw_rows)

    # Write files to target directory
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df_paper_a19.to_csv(out_dir / "table_A19_edge_masking_comparison.csv", index=False)
    df_paper_a19.to_csv(out_dir / "table_A19_performance_summary.csv", index=False)
    df_extended_a19.to_csv(out_dir / "table_A19_extended_variants.csv", index=False)
    df_summary.to_csv(out_dir / "table_A19_extended_summary.csv", index=False)

    df_vs_base_10run.to_csv(out_dir / "masking_vs_baselines_10run.csv", index=False)
    df_vs_base_50fold.to_csv(out_dir / "masking_vs_baselines_50fold.csv", index=False)
    df_ablation_10run.to_csv(out_dir / "masking_ablation_vs_original_10run.csv", index=False)
    df_ablation_50fold.to_csv(out_dir / "masking_ablation_vs_original_50fold.csv", index=False)
    df_raw.to_csv(out_dir / "raw_50fold_matrices.csv", index=False)

    # Save Excel Workbook with professional formatting
    excel_path = out_dir / "Table_A19_Statistical_Validation.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_paper_a19.to_excel(writer, sheet_name="Table_A19_Paper_Format", index=False)
        df_extended_a19.to_excel(writer, sheet_name="Table_A19_Extended_Variants", index=False)
        df_summary.to_excel(writer, sheet_name="Table_A19_Summary", index=False)
        df_vs_base_10run.to_excel(writer, sheet_name="Vs_Baselines_10Run", index=False)
        df_ablation_10run.to_excel(writer, sheet_name="Ablation_vs_Original_10Run", index=False)
        df_vs_base_50fold.to_excel(writer, sheet_name="Vs_Baselines_50Fold", index=False)
        df_ablation_50fold.to_excel(writer, sheet_name="Ablation_vs_Original_50Fold", index=False)
        df_raw.to_excel(writer, sheet_name="Raw_50Fold_Data", index=False)

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F497D")
                cell.alignment = Alignment(horizontal="center", vertical="center")
            for column_cells in sheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells]
                width = min(max(max(map(len, values)) + 3, 12), 45)
                sheet.column_dimensions[column_cells[0].column_letter].width = width

    # Save JSON Archive
    json_path = out_dir / "statistical_summary.json"
    summary_payload = {
        "metadata": {
            "title": "Table A19 Edge-Masking Analysis Statistical Validation (Scheme C)",
            "models": models,
            "datasets": datasets,
            "metrics": metrics,
            "bootstrap_samples": args.bootstrap,
            "seed": args.seed,
            "methodology": {
                "primary_test": "Exact two-sided Wilcoxon signed-rank test on 10-run means",
                "supplementary_test": "Asymptotic two-sided Wilcoxon signed-rank test on 50 fold values",
                "effect_size": "Paired rank-biserial correlation r_rb",
                "location_shift": "Paired Hodges-Lehmann difference (left minus right)",
                "confidence_interval": "95% percentile bootstrap interval (10,000 resamples)",
                "multiple_testing": "Benjamini-Hochberg FDR correction",
            },
        },
        "table_A19_summary": df_summary.to_dict(orient="records"),
        "vs_baselines_10run": df_vs_base_10run.to_dict(orient="records"),
        "ablation_vs_original_10run": df_ablation_10run.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Successfully generated all Scheme C statistical deliverables in: {args.output_dir}")
    print(f"  - Table A19 Summary: {args.output_dir / 'table_A19_extended_summary.csv'}")
    print(f"  - Vs Baselines (10-run): {args.output_dir / 'masking_vs_baselines_10run.csv'}")
    print(f"  - Vs Baselines (50-fold): {args.output_dir / 'masking_vs_baselines_50fold.csv'}")
    print(f"  - Ablation vs Original (10-run): {args.output_dir / 'masking_ablation_vs_original_10run.csv'}")
    print(f"  - Ablation vs Original (50-fold): {args.output_dir / 'masking_ablation_vs_original_50fold.csv'}")
    print(f"  - Raw 50-fold Matrices: {args.output_dir / 'raw_50fold_matrices.csv'}")
    print(f"  - Excel Workbook: {args.output_dir / 'Table_A19_Statistical_Validation.xlsx'}")
    print(f"  - JSON Report: {args.output_dir / 'statistical_summary.json'}")


if __name__ == "__main__":
    main()
