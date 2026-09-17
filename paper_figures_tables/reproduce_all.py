#!/usr/bin/env python3
"""
BIMODriver Statistical Validation & Paper Reproduction Suite
============================================================

This script is the master entrypoint for reproducing all tables, figures,
statistical tests, and Excel workbooks reported in the paper:
"BIMODriver: Decoupling Multimodal Feature Interaction and Structural
Inductive Bias for Cancer Driver Gene Identification"

Paper Deliverables Mapping:
- Table 1: Pan-cancer benchmark on CPDB and STRING networks
- Table 2: 15 Cancer-specific benchmark comparisons
- Table A8 & Table A9: Pan-cancer and cancer-specific significance summaries
- Table A10 & Table A11: Detailed pan-cancer & cancer-specific statistical significance tests
- Table A19: Edge-masking information exposure & strict inductive learning
- Table R5: Label-like phrases masking ablation statistical tests
- Clean <-> Hit Benchmark: Cross-distribution vocabulary leakage evaluation
- Ablation Study: 4-Way protocol and feature ablation
- Figure 2: Independent validation & comparative performance scatter plot
- Figure 3: Component ablation breakdown (Bar chart & Radar chart)
- Figure 4: Sparse Mixture-of-Experts (SMoE) gating visualization
- Figure 5: Candidate cancer driver gene prediction & multi-database validation (Table A12)
- Supplementary Workbook & 50-Fold Sensitivity: Statistical significance workbooks & sensitivity tests

Usage:
  python reproduce_all.py --all           # Run full reproduction pipeline
  python reproduce_all.py --tables        # Reproduce all paper tables & statistical tests
  python reproduce_all.py --figures       # Reproduce Figures 2, 3, 4, 5
  python reproduce_all.py --workbooks     # Re-generate standardized Excel workbooks
  python reproduce_all.py --verify        # Verify existence and integrity of all outputs
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_EXE = os.environ.get("PYTHON_EXE", sys.executable)
if not os.environ.get("PYTHON_EXE"):
    conda_py = Path.home() / ".conda" / "envs" / "bimodriver-statistics" / "bin" / "python"
    if conda_py.exists():
        try:
            import pandas  # noqa: F401
        except ImportError:
            PYTHON_EXE = str(conda_py)


def run_step(description: str, cmd: list[str]) -> bool:
    print(f"\n{'='*75}")
    print(f">> [RUNNING] {description}")
    print(f">> Command: {' '.join(cmd)}")
    print(f"{'='*75}")
    start_time = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT_DIR))
    elapsed = time.time() - start_time
    if result.returncode == 0:
        print(f">> [SUCCESS] {description} ({elapsed:.2f}s)")
        return True
    else:
        print(f">> [FAILED] {description} (exit code: {result.returncode})", file=sys.stderr)
        return False


def reproduce_tables():
    print("\n" + "#"*75)
    print("# STEP 1: PAPER TABLES & STATISTICAL SIGNIFICANCE TESTS")
    print("#"*75)

    success = True

    # 1. Table 1 & Table 2: Pan-cancer and 15 cancer-specific benchmarks
    success &= run_step(
        "Table 1 & Table 2: Pan-cancer & 15 Cancer Benchmarks (reanalyse_statistics.py)",
        [PYTHON_EXE, "scripts/paper_validation/reanalyse_statistics.py"]
    )

    # 2. Clean <-> Hit Vocabulary Leakage Benchmark
    success &= run_step(
        "Clean <-> Hit Vocabulary Leakage Benchmark (run_clean_hit_evaluation.py)",
        [PYTHON_EXE, "scripts/leakage_ablation/run_clean_hit_evaluation.py"]
    )

    # 3. Table A19: Edge-Masking Ablation & Strict Inductive Learning
    success &= run_step(
        "Table A19: Edge-Masking & Strict Inductive Analysis (run_table_A19_statistics.py)",
        [PYTHON_EXE, "scripts/leakage_ablation/run_table_A19_statistics.py"]
    )

    # 4. 4-Way Pan-cancer Feature & Protocol Ablation
    success &= run_step(
        "Pan-Cancer Feature & Protocol Ablation (run_statistics.py)",
        [PYTHON_EXE, "scripts/leakage_ablation/run_statistics.py"]
    )

    # 5. Table R5: Label-Like Phrases Masking Statistical Analysis
    success &= run_step(
        "Table R5: Label-Like Phrases Masking Analysis (run_table_R5_statistics.py)",
        [PYTHON_EXE, "scripts/leakage_ablation/run_table_R5_statistics.py"]
    )

    return success


def reproduce_figures():
    print("\n" + "#"*75)
    print("# STEP 2: PAPER FIGURES (HIGH-RESOLUTION 300 DPI + VECTOR PDF)")
    print("#"*75)

    success = True

    # Figure 2: Independent validation & comparison scatter plot
    success &= run_step(
        "Figure 2: Independent Validation Scatter Plot (plot_figure2_comparison.py)",
        [PYTHON_EXE, "scripts/figures/plot_figure2_comparison.py"]
    )

    # Figure 3: Ablation study component breakdown (Bar chart & Radar chart)
    success &= run_step(
        "Figure 3: Ablation Study Component Breakdown (plot_figure3_ablation.py)",
        [PYTHON_EXE, "scripts/figures/plot_figure3_ablation.py"]
    )

    # Figure 4: SMoE gating visualization
    success &= run_step(
        "Figure 4: SMoE Gating Visualization (plot_figure4_smoe_gating.py)",
        [PYTHON_EXE, "scripts/figures/plot_figure4_smoe_gating.py"]
    )

    # Figure 5: Candidate cancer driver genes multi-database validation
    success &= run_step(
        "Figure 5: Candidate Genes Multi-Database Validation (plot_figure5_candidate_genes.py)",
        [PYTHON_EXE, "scripts/figures/plot_figure5_candidate_genes.py"]
    )

    return success


def reproduce_workbooks():
    print("\n" + "#"*75)
    print("# STEP 3: EXCEL WORKBOOKS VERIFICATION")
    print("#"*75)

    success = True

    # Standardize / verify Statistical Significance Summary Workbook
    success &= run_step(
        "Verify Statistical Significance Summary Workbook (standardize_workbooks.py)",
        [PYTHON_EXE, "scripts/workbooks/standardize_workbooks.py"]
    )

    return success


def verify_outputs():
    print("\n" + "#"*75)
    print("# STEP 4: VERIFICATION OF ARTIFACTS & INTEGRITY AUDIT")
    print("#"*75)

    expected_files = [
        # Paper Main Tables (Strictly aligned with main.tex)
        ("Table 1 Performance Comparison", "results/paper_validation/table1_performance_comparison.csv"),
        ("Table 2 Cancer Specific Comparison", "results/paper_validation/table2_cancer_specific_comparison.csv"),
        ("Table 1 & 2 Summary", "results/paper_validation/benchmark_summary.csv"),
        ("Table 1 & 2 Run Means", "results/paper_validation/benchmark_run_means.csv"),
        ("Table 2 Cancer Type Means", "results/paper_validation/cancer_type_means.csv"),
        ("50-Fold Sensitivity Analysis", "results/paper_validation/sensitivity_fold50.csv"),
        # Appendix Statistical Tables (Strictly aligned with appendix.tex)
        ("Table A8 Pan-Cancer Significance", "results/paper_validation/table_A8_pancancer_significance.csv"),
        ("Table A9 Cancer-Specific Significance", "results/paper_validation/table_A9_cancer_specific_significance.csv"),
        ("Table A10 Pan-Cancer Detailed Significance", "results/paper_validation/table_A10_pancancer_significance.csv"),
        ("Table A11 Cancer-Specific Detailed Significance", "results/paper_validation/table_A11_cancer_specific_significance.csv"),
        # Clean <-> Hit Benchmark outputs (Reviewer response / Appendix)
        ("Clean-Hit Paper Main Table (Two-Sided)", "results/clean_hit_results_final/clean_hit_paper_main_table.csv"),
        ("Clean-Hit Main Table (Two-Sided)", "results/clean_hit_results_final/clean_hit_paper_main_table_two_sided.csv"),
        ("Clean-Hit Supplementary One-Sided Table", "results/clean_hit_results_final/clean_hit_paper_supplementary_one_sided.csv"),
        ("Clean-Hit Statistical Comparison", "results/clean_hit_results_final/clean_hit_statistical_comparison.csv"),
        ("Clean-Hit Descriptive Summary", "results/clean_hit_results_final/clean_hit_descriptive_summary.csv"),
        ("Clean-Hit Raw Run Scores", "results/clean_hit_results_final/clean_hit_raw_scores.csv"),
        ("Clean-Hit Excel Workbook", "results/clean_hit_results_final/Clean_Hit_Statistical_Validation.xlsx"),
        # Table A19 Edge-Masking outputs (Strictly aligned with latest appendix.tex)
        ("Table A19 Edge-Masking Comparison", "results/edge_masking_A19/table_A19_edge_masking_comparison.csv"),
        ("Table A19 Extended Variants", "results/edge_masking_A19/table_A19_extended_variants.csv"),
        ("Table A19 Extended Summary", "results/edge_masking_A19/table_A19_extended_summary.csv"),
        ("Table A19 Masking vs Baselines (50-fold)", "results/edge_masking_A19/masking_vs_baselines_50fold.csv"),
        ("Table A19 Masking vs Baselines (10-run)", "results/edge_masking_A19/masking_vs_baselines_10run.csv"),
        ("Table A19 Masking Ablation (50-fold)", "results/edge_masking_A19/masking_ablation_vs_original_50fold.csv"),
        ("Table A19 Raw 50-Fold Matrices", "results/edge_masking_A19/raw_50fold_matrices.csv"),
        ("Table A19 Excel Workbook", "results/edge_masking_A19/Table_A19_Statistical_Validation.xlsx"),
        # Table R5 Label-Like Phrases Masking outputs
        ("Table R5 Formatted CSV", "results/label_like_phrases_masking/table_R5_label_masking_formatted.csv"),
        ("Table R5 Full Precision CSV", "results/label_like_phrases_masking/table_R5_label_masking_full_precision.csv"),
        ("Table R5 Descriptive Summary", "results/label_like_phrases_masking/table_R5_descriptive_summary.csv"),
        ("Table R5 Run-Level Means", "results/label_like_phrases_masking/table_R5_run_level_means.csv"),
        ("Table R5 Excel Workbook", "results/label_like_phrases_masking/Table_R5_Statistical_Validation.xlsx"),
        # Core Statistical Summary Workbook (Tables 1 & 2 / Tables A8-A11)
        ("Statistical Significance Summary Excel", "results/workbooks/Statistical_Significance_Summary.xlsx"),
        # Paper Figures (High-Resolution 300 DPI PNG + Vector PDF)
        ("Figure 2 PNG", "results/figures/figure2_independent_validation.png"),
        ("Figure 2 PDF", "results/figures/figure2_independent_validation.pdf"),
        ("Figure 3 Bar PNG", "results/figures/figure3_ablation_study.png"),
        ("Figure 3 Bar PDF", "results/figures/figure3_ablation_study.pdf"),
        ("Figure 3 Radar PNG", "results/figures/figure3_ablation_radar.png"),
        ("Figure 4 PNG", "results/figures/figure4_smoe_gating.png"),
        ("Figure 4 PDF", "results/figures/figure4_smoe_gating.pdf"),
        # Figure 5 Subplots & Composite
        ("Figure 5a Database Validation PNG", "results/figures/figure5a_database_validation.png"),
        ("Figure 5a Database Validation PDF", "results/figures/figure5a_database_validation.pdf"),
        ("Figure 5b Venn Diagram PNG", "results/figures/figure5b_venn_diagram.png"),
        ("Figure 5b Venn Diagram PDF", "results/figures/figure5b_venn_diagram.pdf"),
        ("Figure 5c Enrichment Analysis PNG", "results/figures/figure5c_enrichment_analysis.png"),
        ("Figure 5c Enrichment Analysis PDF", "results/figures/figure5c_enrichment_analysis.pdf"),
        ("Figure 5 Composite PNG", "results/figures/figure5_candidate_genes_composite.png"),
        ("Figure 5 Composite PDF", "results/figures/figure5_candidate_genes_composite.pdf"),
    ]

    all_ok = True
    print(f"{'Artifact Description':<40} | {'Status':<8} | {'Size':>10} | {'Path'}")
    print("-" * 95)

    for desc, rel_path in expected_files:
        full_path = ROOT_DIR / rel_path
        if full_path.exists():
            size = full_path.stat().st_size
            if size > 0:
                print(f"{desc:<40} | PASS     | {size:>10,} B | {rel_path}")
            else:
                print(f"{desc:<40} | EMPTY    | {size:>10,} B | {rel_path}", file=sys.stderr)
                all_ok = False
        else:
            print(f"{desc:<40} | MISSING  | {'N/A':>10} | {rel_path}", file=sys.stderr)
            all_ok = False

    print("-" * 95)
    if all_ok:
        print(f"[AUDIT SUCCESS] All {len(expected_files)} expected paper deliverables verified successfully!\n")
    else:
        print("[AUDIT WARNING] Some deliverables are missing or empty.\n", file=sys.stderr)
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="BIMODriver Paper Reproduction Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--all", action="store_true", help="Run entire reproduction pipeline (tables, figures, workbooks, verify)")
    parser.add_argument("--tables", action="store_true", help="Reproduce Tables 1, 2, A8-A11, Clean-Hit, A19 and ablation tests")
    parser.add_argument("--figures", action="store_true", help="Reproduce Figures 2, 3, 4, 5")
    parser.add_argument("--workbooks", action="store_true", help="Re-export and standardize Excel workbooks")
    parser.add_argument("--verify", action="store_true", help="Audit and verify generated artifacts")

    args = parser.parse_args()

    # Default to running everything if no flags provided
    if not any([args.all, args.tables, args.figures, args.workbooks, args.verify]):
        args.all = True

    start_total = time.time()
    success = True

    if args.all or args.tables:
        success &= reproduce_tables()

    if args.all or args.figures:
        success &= reproduce_figures()

    if args.all or args.workbooks:
        success &= reproduce_workbooks()

    if args.all or args.verify:
        success &= verify_outputs()

    elapsed_total = time.time() - start_total
    print(f"\nReproduction pipeline completed in {elapsed_total:.2f}s.")
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
