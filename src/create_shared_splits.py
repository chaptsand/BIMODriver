import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from alignment_check import assert_data_alignment

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_PATH = os.path.join(BASE_DIR, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')
SPLIT_OUT_PATH = os.path.join(BASE_DIR, 'data', 'CPDB', 'leakage_splits_10runs.pkl')

def create_and_verify_splits():
    # 0. 先行执行全局数据与特征一致性断言检查
    assert_data_alignment(BASE_DIR)

    print(f"Reading audit data from: {AUDIT_PATH}")
    df = pd.read_excel(AUDIT_PATH, sheet_name='Gene-level Flags')
    
    # 1. 严格检查与排序
    df = df.sort_values('Code_Index').reset_index(drop=True)
    assert len(df) == 13627, f"Expected 13627 genes, got {len(df)}"
    assert (df['Code_Index'].values == np.arange(13627)).all(), "Code_Index must be continuous 0..13626"
    
    flag_col = 'LLM combined | Any exact label-like term'
    labeled_mask = (df['Label_Status'] == 'Labeled').values
    hit_flag = (df[flag_col] == 1).values
    
    clean_mask = labeled_mask & (~hit_flag)
    hit_mask = labeled_mask & hit_flag
    
    # 类别标签
    gene_labels = (df['Gene_Label'] == 'Driver').values.astype(int)
    
    # 断言样本量与类别分布
    assert labeled_mask.sum() == 2983, f"Expected 2983 labeled, got {labeled_mask.sum()}"
    assert clean_mask.sum() == 2402, f"Expected 2402 clean, got {clean_mask.sum()}"
    assert hit_mask.sum() == 581, f"Expected 581 hit, got {hit_mask.sum()}"
    
    clean_drivers = (gene_labels[clean_mask] == 1).sum()
    clean_nondrivers = (gene_labels[clean_mask] == 0).sum()
    assert clean_drivers == 571 and clean_nondrivers == 1831, f"Clean counts mismatch: drivers={clean_drivers}, nondrivers={clean_nondrivers}"
    
    hit_drivers = (gene_labels[hit_mask] == 1).sum()
    hit_nondrivers = (gene_labels[hit_mask] == 0).sum()
    assert hit_drivers == 225 and hit_nondrivers == 356, f"Hit counts mismatch: drivers={hit_drivers}, nondrivers={hit_nondrivers}"
    
    splits_dict = {
        'clean_to_hit': [],
        'hit_to_clean': [],
        'metadata': {
            'total_genes': 13627,
            'labeled_genes': 2983,
            'clean_count': 2402,
            'hit_count': 581,
            'flag_col': flag_col,
            'n_runs': 10,
            'base_seed': 42,
            'val_ratio': 0.2
        }
    }
    
    # 2. 生成 clean_to_hit 10 runs
    clean_indices = np.where(clean_mask)[0]
    clean_strat_labels = gene_labels[clean_indices]
    fixed_hit_indices = np.where(hit_mask)[0]
    
    print("\nGenerating clean_to_hit splits (10 runs)...")
    for exp_id in range(10):
        seed = 42 + exp_id
        tr_idx, val_idx = train_test_split(
            clean_indices,
            test_size=0.2,
            stratify=clean_strat_labels,
            random_state=seed
        )
        tr_idx = np.sort(tr_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(fixed_hit_indices)
        
        # 严格断言
        assert len(tr_idx) == 1921, f"Expected 1921 train, got {len(tr_idx)}"
        assert len(val_idx) == 481, f"Expected 481 val, got {len(val_idx)}"
        assert len(test_idx) == 581, f"Expected 581 test, got {len(test_idx)}"
        
        # 无交集断言
        assert len(set(tr_idx) & set(val_idx)) == 0, "Train and Val overlap!"
        assert len(set(tr_idx) & set(test_idx)) == 0, "Train and Test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val and Test overlap!"
        assert len(set(tr_idx) | set(val_idx)) == 2402, "Train + Val does not cover clean candidate pool!"
        
        # 比例检查
        tr_dr = (gene_labels[tr_idx] == 1).sum()
        tr_non = (gene_labels[tr_idx] == 0).sum()
        val_dr = (gene_labels[val_idx] == 1).sum()
        val_non = (gene_labels[val_idx] == 0).sum()
        assert tr_dr == 457 and tr_non == 1464, f"Run {exp_id} train class mismatch: {tr_dr}, {tr_non}"
        assert val_dr == 114 and val_non == 367, f"Run {exp_id} val class mismatch: {val_dr}, {val_non}"
        
        splits_dict['clean_to_hit'].append({
            'exp_id': exp_id,
            'seed': seed,
            'train_idx': tr_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })
    print("  clean_to_hit: All 10 runs generated and validated successfully.")
    
    # 3. 生成 hit_to_clean 10 runs
    hit_indices = np.where(hit_mask)[0]
    hit_strat_labels = gene_labels[hit_indices]
    fixed_clean_indices = np.where(clean_mask)[0]
    
    print("\nGenerating hit_to_clean splits (10 runs)...")
    for exp_id in range(10):
        seed = 42 + exp_id
        tr_idx, val_idx = train_test_split(
            hit_indices,
            test_size=0.2,
            stratify=hit_strat_labels,
            random_state=seed
        )
        tr_idx = np.sort(tr_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(fixed_clean_indices)
        
        # 严格断言
        assert len(tr_idx) == 464, f"Expected 464 train, got {len(tr_idx)}"
        assert len(val_idx) == 117, f"Expected 117 val, got {len(val_idx)}"
        assert len(test_idx) == 2402, f"Expected 2402 test, got {len(test_idx)}"
        
        # 无交集断言
        assert len(set(tr_idx) & set(val_idx)) == 0, "Train and Val overlap!"
        assert len(set(tr_idx) & set(test_idx)) == 0, "Train and Test overlap!"
        assert len(set(val_idx) & set(test_idx)) == 0, "Val and Test overlap!"
        assert len(set(tr_idx) | set(val_idx)) == 581, "Train + Val does not cover hit candidate pool!"
        
        # 比例检查
        tr_dr = (gene_labels[tr_idx] == 1).sum()
        tr_non = (gene_labels[tr_idx] == 0).sum()
        val_dr = (gene_labels[val_idx] == 1).sum()
        val_non = (gene_labels[val_idx] == 0).sum()
        assert tr_dr == 180 and tr_non == 284, f"Run {exp_id} train class mismatch: {tr_dr}, {tr_non}"
        assert val_dr == 45 and val_non == 72, f"Run {exp_id} val class mismatch: {val_dr}, {val_non}"
        
        splits_dict['hit_to_clean'].append({
            'exp_id': exp_id,
            'seed': seed,
            'train_idx': tr_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        })
    print("  hit_to_clean: All 10 runs generated and validated successfully.")
    
    # 4. 保存公共 split 文件
    os.makedirs(os.path.dirname(SPLIT_OUT_PATH), exist_ok=True)
    with open(SPLIT_OUT_PATH, 'wb') as f:
        pickle.dump(splits_dict, f)
    print(f"\nSaved public split file to: {SPLIT_OUT_PATH} (size: {os.path.getsize(SPLIT_OUT_PATH)} bytes)")
    
    # 5. 重新读取验证
    with open(SPLIT_OUT_PATH, 'rb') as f:
        loaded = pickle.load(f)
    assert len(loaded['clean_to_hit']) == 10
    assert len(loaded['hit_to_clean']) == 10
    print("Verification of saved split file succeeded!")

if __name__ == '__main__':
    create_and_verify_splits()
