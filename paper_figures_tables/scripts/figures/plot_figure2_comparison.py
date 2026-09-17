#!/usr/bin/env python3
"""
Figure 2: Independent Validation & Comparative Performance Scatter Plot
Paper Reference: Figure 2 (Comparison of AUPRC performance on Oncogene and OncoKB datasets)

This script plots the cross-validation and independent test benchmark comparison
between BIMODriver and 9 competitive baseline methods:
- Baselines: EMOGI, ECD-CDGI, MTGCN, MNGCL, DISFusion, DISHyper, GCN, GAT, ChebNet
- Proposed: BIMODriver
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_figure2(output_dir: str = "results/figures", formats=("png", "pdf")):
    os.makedirs(output_dir, exist_ok=True)

    # Style settings
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'legend.fontsize': 13,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    methods = [
        "EMOGI", "ECD-CDGI", "MTGCN", "MNGCL", "DISFusion",
        "DISHyper", "GCN", "GAT", "ChebNet", "BIMODriver"
    ]
    # AUPRC on Oncogene (x-axis) vs AUPRC on OncoKB (y-axis)
    x_values = [0.0987, 0.0996, 0.1016, 0.1018, 0.1070, 0.0997, 0.0881, 0.0755, 0.0957, 0.1212]
    y_values = [0.0772, 0.0799, 0.0818, 0.0829, 0.1075, 0.0958, 0.0749, 0.0625, 0.0795, 0.1223]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
    ax.set_facecolor('white')
    ax.grid(True, linestyle='--', alpha=0.5, color='lightgray', linewidth=1.2)

    markers = ['o', 's', '^', 'v', 'D', 'p', 'h', 'P', 'X', '*']
    colors = matplotlib.colormaps['tab10'](np.linspace(0, 1, len(methods)))

    for i, (m, x, y) in enumerate(zip(methods, x_values, y_values)):
        if m == "BIMODriver":
            # Highlight BIMODriver with a prominent star marker
            ax.scatter(x, y, marker='*', color='#D62728', s=650,
                       edgecolors='#8B0000', linewidths=2.5, label=m, zorder=10)
            ax.scatter(x, y, marker='*', facecolors='none',
                       edgecolors='white', linewidths=1.5, s=700, zorder=9)
            ax.annotate(f"{m}\n({x:.4f}, {y:.4f})",
                        xy=(x, y), xytext=(x - 0.003, y + 0.004),
                        fontsize=12, fontweight='bold', color='#8B0000',
                        arrowprops=dict(arrowstyle="->", color='#8B0000', lw=1.2))
        else:
            ax.scatter(x, y, marker=markers[i], color=colors[i], s=160,
                       edgecolors='dimgray', linewidths=0.8, label=m, zorder=5, alpha=0.9)
            # Label baseline points
            ax.annotate(m, xy=(x, y), xytext=(x + 0.0012, y - 0.0015),
                        fontsize=9, color='#333333', alpha=0.85)

    # Reference threshold lines
    ax.axvline(x=0.12, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.axhline(y=0.12, color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
    ax.text(0.1205, 0.062, 'High Performance Zone (AUPRC > 0.12)', color='gray',
            fontsize=10, fontstyle='italic')

    ax.set_xlabel('AUPRC (Oncogene Dataset)', fontsize=15, fontweight='bold', color='#333333')
    ax.set_ylabel('AUPRC (OncoKB Dataset)', fontsize=15, fontweight='bold', color='#333333')
    ax.set_title('Figure 2: Independent Validation & Comparative Performance',
                 fontsize=16, fontweight='bold', color='#222222', pad=15)

    ax.set_xlim(0.07, 0.13)
    ax.set_ylim(0.058, 0.13)

    ax.tick_params(axis='both', direction='inout', length=6, width=1.2, colors='black')

    ax.legend(loc='lower right', ncol=2, fontsize=10.5,
              frameon=True, framealpha=0.95, edgecolor='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()

    out_paths = []
    for fmt in formats:
        p = os.path.join(output_dir, f"figure2_independent_validation.{fmt}")
        plt.savefig(p, dpi=300, bbox_inches='tight')
        out_paths.append(p)

    plt.close()
    for p in out_paths:
        print(f"[Figure 2] Saved to: {p}")
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paper Figure 2")
    parser.add_argument("--output_dir", default="results/figures", help="Output directory")
    args = parser.parse_args()
    plot_figure2(args.output_dir)
