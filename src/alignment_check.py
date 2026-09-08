import os
import torch
import numpy as np
import pandas as pd

def assert_data_alignment(base_dir=None):
    """
    严格数据对齐断言：
    1. 审计表 (Gemma_Vocabulary_Leakage_Audit.xlsx)、CPDB 索引文件 (CPDB_gene_index_id_name_label.csv)、
       node_names.txt、以及 DISFusion geneList.txt 的基因总数均为 13,627，且基因顺序逐行 100% 一致。
    2. BIMODriver/CPDB (CPDB_new_data.pt)、MNGCL (CPDB_data.pkl) 和 DISFusion (biological features.csv)
       所使用的 48 维生物多组学特征数值完全一致 (atol=1e-5)。
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    audit_path = os.path.join(base_dir, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')
    cpdb_csv_path = os.path.join(base_dir, 'src', 'CPDB_gene_index_id_name_label.csv')
    dis_gene_path = os.path.join(base_dir, 'data', 'DISFusion_data', 'geneList.txt')
    node_names_path = os.path.join(base_dir, 'data', 'CPDB', 'node_names.txt')

    assert os.path.exists(audit_path), f"Audit file not found: {audit_path}"
    assert os.path.exists(cpdb_csv_path), f"CPDB index file not found: {cpdb_csv_path}"
    assert os.path.exists(dis_gene_path), f"DISFusion geneList not found: {dis_gene_path}"
    assert os.path.exists(node_names_path), f"Node names file not found: {node_names_path}"

    audit_df = pd.read_excel(audit_path, sheet_name='Gene-level Flags').sort_values('Code_Index').reset_index(drop=True)
    cpdb_df = pd.read_csv(cpdb_csv_path).sort_values('Code_Index').reset_index(drop=True)
    dis_gene_list = pd.read_csv(dis_gene_path, header=None).iloc[:, 1].values

    with open(node_names_path, 'r', encoding='utf-8') as f:
        node_names = [line.strip().split(',')[1] for line in f if line.strip()]

    # 1. 基因数量检查
    assert len(audit_df) == 13627, f"Audit sheet length mismatch: {len(audit_df)}"
    assert len(cpdb_df) == 13627, f"CPDB index csv length mismatch: {len(cpdb_df)}"
    assert len(dis_gene_list) == 13627, f"DISFusion geneList length mismatch: {len(dis_gene_list)}"
    assert len(node_names) == 13627, f"Node names length mismatch: {len(node_names)}"

    # 2. Code_Index 连续性检查
    assert (audit_df['Code_Index'].values == np.arange(13627)).all(), "Audit Code_Index not continuous 0..13626"
    assert (cpdb_df['Code_Index'].values == np.arange(13627)).all(), "CPDB Code_Index not continuous 0..13626"

    # 3. 基因名称逐行严格一致性断言
    assert np.array_equal(audit_df['Gene_Name'].values, cpdb_df['Gene_Name'].values), \
        "Audit sheet and CPDB index CSV gene names do not match!"
    assert np.array_equal(audit_df['Gene_Name'].values, dis_gene_list), \
        "Audit sheet and DISFusion geneList.txt gene names do not match!"
    assert np.array_equal(audit_df['Gene_Name'].values, node_names), \
        "Audit sheet and CPDB node_names.txt gene names do not match!"

    # 4. 48 维生物特征数值逐行严格一致性断言
    cpdb_pt_path = os.path.join(base_dir, 'data', 'CPDB', 'CPDB_new_data.pt')
    mngcl_pkl_path = os.path.join(base_dir, 'data', 'CPDB', 'CPDB_data.pkl')
    dis_feat_path = os.path.join(base_dir, 'data', 'DISFusion_data', 'biological features.csv')

    assert os.path.exists(cpdb_pt_path), f"CPDB_new_data.pt not found: {cpdb_pt_path}"
    assert os.path.exists(mngcl_pkl_path), f"CPDB_data.pkl not found: {mngcl_pkl_path}"
    assert os.path.exists(dis_feat_path), f"biological features.csv not found: {dis_feat_path}"

    pt_data = torch.load(cpdb_pt_path, map_location='cpu')
    feat_cpdb = pt_data.x[:, :48].numpy()

    mngcl_data = torch.load(mngcl_pkl_path, map_location='cpu')
    feat_mngcl = mngcl_data.x[:, :48].numpy()

    feat_dis = pd.read_csv(dis_feat_path, sep=',').values

    assert feat_cpdb.shape == (13627, 48), f"feat_cpdb shape: {feat_cpdb.shape}"
    assert feat_mngcl.shape == (13627, 48), f"feat_mngcl shape: {feat_mngcl.shape}"
    assert feat_dis.shape == (13627, 48), f"feat_dis shape: {feat_dis.shape}"

    max_diff_cpdb_mngcl = np.max(np.abs(feat_cpdb - feat_mngcl))
    max_diff_cpdb_dis = np.max(np.abs(feat_cpdb - feat_dis))
    assert max_diff_cpdb_mngcl < 1e-4, f"CPDB vs MNGCL 48D max diff too large: {max_diff_cpdb_mngcl}"
    assert max_diff_cpdb_dis < 1e-4, f"CPDB vs DISFusion 48D max diff too large: {max_diff_cpdb_dis}"

    print(f"[Assertion Passed] All 13,627 genes and 48-dim features across Audit, CPDB, MNGCL, and DISFusion are 100% aligned!")

if __name__ == '__main__':
    assert_data_alignment()
