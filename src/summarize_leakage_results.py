import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BASE_DIR, 'result')

def parse_summary_file(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 严格在 Final Fixed Test Metrics 区块提取最终测试集评估指标，避免误匹配验证集指标
    test_sec = content
    if "Final Fixed Test Metrics" in content:
        test_sec = content.split("Final Fixed Test Metrics")[1]

    auroc_match = re.search(r'(?:Test\s+)?AUROC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)', test_sec)
    auprc_match = re.search(r'(?:Test\s+)?AUPRC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)', test_sec)

    if not auroc_match or not auprc_match:
        return None

    auroc = float(auroc_match.group(1))
    auroc_std = float(auroc_match.group(2))
    auprc = float(auprc_match.group(1))
    auprc_std = float(auprc_match.group(2))

    return {
        'auroc': auroc,
        'auroc_std': auroc_std,
        'auprc': auprc,
        'auprc_std': auprc_std,
        'formatted': f"{auroc:.4f} ± {auroc_std:.4f} / {auprc:.4f} ± {auprc_std:.4f}"
    }

def print_table(title, subtitle, methods, splits):
    print("\n" + "=" * 105)
    print(title)
    if subtitle:
        print(subtitle)
    print("=" * 105)

    rows = []
    for m_name, m_prefix in methods:
        row = {'Method': m_name}
        for s_label, s_name in splits:
            path = os.path.join(RES_DIR, f"{m_prefix}_leakage_{s_name}_summary.txt")
            res = parse_summary_file(path)
            tag = ""
            if not res:
                # 检查 smoke test 备用
                smoke_path = os.path.join(RES_DIR, f"smoke_{m_prefix}_leakage_{s_name}_summary.txt")
                res = parse_summary_file(smoke_path)
                if res:
                    tag = " [Smoke]"
            if res:
                row[f"{s_label} AUROC"] = f"{res['auroc']:.4f} ± {res['auroc_std']:.4f}{tag}"
                row[f"{s_label} AUPRC"] = f"{res['auprc']:.4f} ± {res['auprc_std']:.4f}{tag}"
            else:
                row[f"{s_label} AUROC"] = "Pending..."
                row[f"{s_label} AUPRC"] = "Pending..."
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("=" * 105)
    return df

def print_comparison_table(trans_methods, induct_methods, splits):
    print("\n" + "=" * 120)
    print("Side-by-Side Comparison: Transductive (传导式) vs. Strict Inductive (严格归纳式)")
    print("Metrics shown as: AUROC / AUPRC (Test Set)")
    print("=" * 120)

    rows = []
    for (m_name, trans_prefix), (_, ind_prefix) in zip(trans_methods, induct_methods):
        display_name = m_name.split(' (')[0]
        row = {'Method': display_name}
        for s_label, s_name in splits:
            trans_path = os.path.join(RES_DIR, f"{trans_prefix}_leakage_{s_name}_summary.txt")
            ind_path = os.path.join(RES_DIR, f"{ind_prefix}_leakage_{s_name}_summary.txt")

            trans_res = parse_summary_file(trans_path) or parse_summary_file(os.path.join(RES_DIR, f"smoke_{trans_prefix}_leakage_{s_name}_summary.txt"))
            ind_res = parse_summary_file(ind_path) or parse_summary_file(os.path.join(RES_DIR, f"smoke_{ind_prefix}_leakage_{s_name}_summary.txt"))

            row[f"{s_label} [Transductive]"] = trans_res['formatted'] if trans_res else "Pending..."
            row[f"{s_label} [Inductive]"] = ind_res['formatted'] if ind_res else "Pending..."

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")
    return df

def main():
    splits = [('Clean -> Hit', 'clean_to_hit'), ('Hit -> Clean', 'hit_to_clean')]

    methods_transductive = [
        ('BIMODriver (Original, Transductive)', 'bimodriver_original'),
        ('DISFusion (Transductive, 64-dim, 10362 hyperedges)', 'disfusion'),
        ('MNGCL (Transductive, 64-dim, 2-view PPI+GO)', 'mngcl')
    ]

    methods_inductive = [
        ('BIMODriver (Original, Inductive)', 'bimodriver_original_inductive'),
        ('DISFusion (Strict Inductive, 64-dim, 10362 hyperedges)', 'disfusion_inductive'),
        ('MNGCL (Strict Inductive, 64-dim, 2-view PPI+GO)', 'mngcl_inductive')
    ]

    # 表 1：传导式评估
    print_table(
        title="Table 1: Transductive Benchmark on Clean <-> Hit Splits (10 Runs)",
        subtitle="Protocol: Full graph message passing during training, Best Checkpoint via Val AUPRC, Single Test Evaluation",
        methods=methods_transductive,
        splits=splits
    )

    # 表 2：严格归纳式评估
    print_table(
        title="Table 2: Strict Inductive Benchmark on Clean <-> Hit Splits (10 Runs)",
        subtitle="Protocol: Test edges cut & test features zeroed during training, Best Checkpoint via Val AUPRC, Full Graph Restored at Eval",
        methods=methods_inductive,
        splits=splits
    )

    # 表 3：传导式 vs 归纳式综合对照
    print_comparison_table(methods_transductive, methods_inductive, splits)

if __name__ == '__main__':
    main()
