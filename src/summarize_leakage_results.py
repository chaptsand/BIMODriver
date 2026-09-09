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

    auroc_match = re.search(r'Test AUROC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)|AUROC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)', content)
    auprc_match = re.search(r'Test AUPRC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)|AUPRC\s*:\s*([\d\.]+)\s*±\s*([\d\.]+)', content)

    if not auroc_match or not auprc_match:
        return None

    auroc = float(auroc_match.group(1) or auroc_match.group(3))
    auroc_std = float(auroc_match.group(2) or auroc_match.group(4))
    auprc = float(auprc_match.group(1) or auprc_match.group(3))
    auprc_std = float(auprc_match.group(2) or auprc_match.group(4))

    return {
        'auroc': auroc,
        'auroc_std': auroc_std,
        'auprc': auprc,
        'auprc_std': auprc_std,
        'formatted': f"{auroc:.4f} ± {auroc_std:.4f} / {auprc:.4f} ± {auprc_std:.4f}"
    }

def main():
    methods = [
        ('BIMODriver (Original)', 'bimodriver_original'),
        ('DISFusion (Paper Setting: 64-dim, 10362 hyperedges)', 'disfusion'),
        ('MNGCL (Paper Setting: 64-dim, 2-view PPI+GO)', 'mngcl')
    ]
    splits = [('Clean -> Hit', 'clean_to_hit'), ('Hit -> Clean', 'hit_to_clean')]

    print("\n" + "=" * 95)
    print("Clean <-> Hit Fixed Split Benchmark Summary (10 Runs)")
    print("Protocol: Model checkpoint selected ONLY on validation set, test set evaluated ONCE at end")
    print("=" * 95)

    rows = []
    for m_name, m_prefix in methods:
        row = {'Method': m_name}
        for s_label, s_name in splits:
            path = os.path.join(RES_DIR, f"{m_prefix}_leakage_{s_name}_summary.txt")
            res = parse_summary_file(path)
            if res:
                row[f"{s_label} (AUROC)"] = f"{res['auroc']:.4f} ± {res['auroc_std']:.4f}"
                row[f"{s_label} (AUPRC)"] = f"{res['auprc']:.4f} ± {res['auprc_std']:.4f}"
            else:
                row[f"{s_label} (AUROC)"] = "Pending..."
                row[f"{s_label} (AUPRC)"] = "Pending..."
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("=" * 95 + "\n")

if __name__ == '__main__':
    main()
