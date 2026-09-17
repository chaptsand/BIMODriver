#!/usr/bin/env python3
"""
Figure 3: Ablation Study Performance Comparison (Bar Chart & Radar Chart)
Paper Reference: Figure 3 (Performance comparison across architectural & feature ablation variants)

Evaluated configurations:
1. Only Context Features
2. Only Omics Feature
3. Only Interactive Features
4. Omics + Context Features
5. Without LLM
6. Omics + Self Features
7. Without SMoE
8. All (Full BIMODriver)
"""

import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke


def plot_figure3_barchart(output_dir: str = "results/figures", formats=("png", "pdf")):
    os.makedirs(output_dir, exist_ok=True)

    methods = [
        "Only Context\nFeatures",
        "Only Omics\nFeatures",
        "Only Interactive\nFeatures",
        "Omics+Context\nFeatures",
        "Without\nLLM",
        "Omics+Self\nFeatures",
        "Without\nSMoE",
        "All (Full\nBIMODriver)"
    ]

    auc_means = [0.8798, 0.9086, 0.9255, 0.9219, 0.9071, 0.9265, 0.9070, 0.9300]
    auprc_means = [0.7554, 0.8272, 0.8486, 0.8466, 0.8222, 0.8466, 0.8234, 0.8587]

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 11,
    })

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='white')
    ax.set_facecolor('white')
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='lightgray', linewidth=1.0)

    n_groups = len(methods)
    bar_width = 0.35
    indices = np.arange(n_groups)

    # Distinct color palette (AUC = warm red/coral, AUPRC = steel blue)
    color_auc = '#E64B35'
    color_auprc = '#3C5488'
    color_auc_highlight = '#B22222'
    color_auprc_highlight = '#1C2841'

    auc_colors = [color_auc if i < n_groups - 1 else color_auc_highlight for i in range(n_groups)]
    auprc_colors = [color_auprc if i < n_groups - 1 else color_auprc_highlight for i in range(n_groups)]

    bars_auc = ax.bar(indices - bar_width / 2, auc_means, bar_width, label='AUC',
                      color=auc_colors, alpha=0.9, edgecolor='black', linewidth=0.8)
    bars_auprc = ax.bar(indices + bar_width / 2, auprc_means, bar_width, label='AUPRC',
                        color=auprc_colors, alpha=0.9, edgecolor='black', linewidth=0.8)

    # Annotate values
    for bar in bars_auc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color_auc)

    for bar in bars_auprc:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}",
                ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=color_auprc)

    # Reference lines for the best scores (Full Model)
    best_auc = max(auc_means)
    best_auprc = max(auprc_means)
    ax.axhline(y=best_auc, color=color_auc, linestyle='-.', alpha=0.6, linewidth=1.2)
    ax.axhline(y=best_auprc, color=color_auprc, linestyle='-.', alpha=0.6, linewidth=1.2)

    ax.set_xticks(indices)
    ax.set_xticklabels(methods, fontweight='medium')
    ax.set_ylim(0.70, 0.98)
    ax.set_ylabel('Metric Score', fontsize=14, fontweight='bold')
    ax.set_title('Figure 3: Ablation Study - Component Breakdown (AUC & AUPRC)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, framealpha=0.95, edgecolor='gray')

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()

    out_paths = []
    for fmt in formats:
        p = os.path.join(output_dir, f"figure3_ablation_study.{fmt}")
        plt.savefig(p, dpi=300, bbox_inches='tight')
        out_paths.append(p)
    plt.close()

    for p in out_paths:
        print(f"[Figure 3 Bar] Saved to: {p}")
    return out_paths


def plot_figure3_radar(output_dir: str = "results/figures", formats=("png", "pdf")):
    os.makedirs(output_dir, exist_ok=True)

    methods_left = [
        "Only Context\nFeatures",
        "Only Omics \nFeature",
        "Only Interactive\nFeatures",
    ]
    methods_right = [
        "Omics+Context\nFeatures",
        "Without\nLLM",
        "Omics+Self\nfeatures",
        "Without\nSMoE"
    ]
    methods = methods_left + ["All (Full)"] + methods_right

    auc_original = [0.8798, 0.9086, 0.9255, 0.9219, 0.9071, 0.9265, 0.9070, 0.9300]
    auprc_original = [0.7554, 0.8272, 0.8486, 0.8466, 0.8222, 0.8466, 0.8234, 0.8587]

    auc_means = [auc_original[0], auc_original[1], auc_original[2], auc_original[7],
                 auc_original[3], auc_original[4], auc_original[5], auc_original[6]]
    auprc_means = [auprc_original[0], auprc_original[1], auprc_original[2], auprc_original[7],
                   auprc_original[3], auprc_original[4], auprc_original[5], auprc_original[6]]

    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'font.weight': 'bold'
    })

    fig = plt.figure(figsize=(11, 11), facecolor='white')
    ax = fig.add_subplot(111, polar=True)

    n_methods = len(methods)
    angles = np.linspace(0, 2 * np.pi, n_methods, endpoint=False)

    all_index = methods.index("All (Full)")
    rotation_angle = np.pi / 2 - angles[all_index]
    angles = (angles + rotation_angle) % (2 * np.pi)

    angles = angles.tolist()
    angles += angles[:1]

    auc_data = auc_means + [auc_means[0]]
    auprc_data = auprc_means + [auprc_means[0]]

    color_auc = '#E64B35'
    color_auprc = '#3C5488'

    ax.plot(angles, auc_data, 'o-', linewidth=2.5, label='AUC', color=color_auc, markersize=8)
    ax.fill(angles, auc_data, alpha=0.25, color=color_auc)
    ax.plot(angles, auprc_data, 's-', linewidth=2.5, label='AUPRC', color=color_auprc, markersize=8)
    ax.fill(angles, auprc_data, alpha=0.25, color=color_auprc)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods, fontsize=12, fontweight='bold')
    ax.set_rlabel_position(30)
    ax.set_ylim(0.70, 0.96)
    ax.set_yticks(np.arange(0.70, 1.00, 0.05))
    ax.tick_params(axis='y', colors='dimgrey', labelsize=10)

    ax.set_facecolor('white')
    ax.spines['polar'].set_visible(False)
    ax.yaxis.grid(linestyle='--', linewidth=0.8, alpha=0.7, color='lightgray')
    ax.xaxis.grid(linestyle='--', linewidth=0.8, alpha=0.7, color='lightgray')

    outline = withStroke(linewidth=2.5, foreground="white")
    for i, (angle, auc_val, auprc_val) in enumerate(zip(angles[:-1], auc_means, auprc_means)):
        r_offset = 0.012
        ax.text(angle, auc_val + r_offset, f'{auc_val:.4f}',
                ha='center', va='bottom', fontsize=10, color=color_auc,
                fontweight='bold', path_effects=[outline])
        ax.text(angle, auprc_val - r_offset * 1.5, f'{auprc_val:.4f}',
                ha='center', va='top', fontsize=10, color=color_auprc,
                fontweight='bold', path_effects=[outline])

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.08), fontsize=12)
    plt.title('Figure 3: Ablation Study (Radar View)', size=18, fontweight='bold', y=1.08, color='#222222')

    plt.tight_layout()
    out_paths = []
    for fmt in formats:
        p = os.path.join(output_dir, f"figure3_ablation_radar.{fmt}")
        plt.savefig(p, dpi=300, bbox_inches='tight')
        out_paths.append(p)
    plt.close()

    for p in out_paths:
        print(f"[Figure 3 Radar] Saved to: {p}")
    return out_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Paper Figure 3")
    parser.add_argument("--output_dir", default="results/figures", help="Output directory")
    args = parser.parse_args()
    plot_figure3_barchart(args.output_dir)
    plot_figure3_radar(args.output_dir)
