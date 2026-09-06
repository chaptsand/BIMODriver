# -*- coding: utf-8 -*-
import os
import sys
import copy
import time
import pickle
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, dropout_adj, negative_sampling, remove_self_loops

import gcnPreprocessing
from model import combine_net_gate_without_ac

device = torch.device('cuda' if torch.cuda.is_available() and os.environ.get('DEVICE') != 'cpu' else 'cpu')
warnings.filterwarnings("ignore")

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "result")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "single"), exist_ok=True)


def save_results_to_file(auroc, auprc, cancerType, dataset='cpdb', lr=0.001, dropout=0.2, lambdinter=0.005):
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    if cancerType == 'pan-cancer':
        if dataset == 'cpdb':
            path = os.path.join(res_dir, 'pan-cancer.txt')
        elif dataset == 'string':
            path = os.path.join(res_dir, 'pan-cancer_string.txt')
        else:
            raise ValueError("Unsupported dataset for pan-cancer results.")
    else:
        single_dir = os.path.join(res_dir, 'single')
        os.makedirs(single_dir, exist_ok=True)
        path = os.path.join(single_dir, 'single.txt')

    with open(path, 'a', encoding='utf-8') as f:
        f.write('--' * 20 + '\n')
        f.write(f"Dropout Rate: {dropout}, Learning Rate: {lr}, Lambda Inter: {lambdinter}\n")
        f.write(f"Results for {cancerType}:\n")
        f.write(f"AUPR: {auroc.mean():.4f} ± {auroc.std():.4f}\n")
        f.write(str(auroc))
        f.write("\n")
        f.write(f"AUC: {auprc.mean():.4f} ± {auprc.std():.4f}\n")
        f.write(str(auprc))
        f.write("\n")


def load_label_single(cancerType):
    path = os.path.join(DATA_DIR, "CPDB", "Specific cancer") + "/"
    label = np.loadtxt(path + "label_file-P-" + cancerType + ".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.loadtxt(path + "pos-" + cancerType + ".txt", dtype=int)
    label_neg = np.loadtxt(path + "neg.txt", dtype=int)
    return Y, label_pos, label_neg


def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    pos_val = pos_label[i * l1:(i + 1) * l1]
    pos_train = list(set(pos_label) - set(pos_val))
    neg_val = neg_label[i * l2:(i + 1) * l2]
    neg_train = list(set(neg_label) - set(neg_val))
    indexs1 = [False] * l
    indexs2 = [False] * l
    for j in range(len(pos_train)):
        indexs1[pos_train[j]] = True
    for j in range(len(neg_train)):
        indexs1[neg_train[j]] = True
    for j in range(len(pos_val)):
        indexs2[pos_val[j]] = True
    for j in range(len(neg_val)):
        indexs2[neg_val[j]] = True
    tr_mask = torch.from_numpy(np.array(indexs1))
    val_mask = torch.from_numpy(np.array(indexs2))
    return tr_mask, val_mask


def get_class_weights(labels):
    pos_counts = labels.sum(dim=0)
    neg_counts = labels.shape[0] - pos_counts
    weights = (neg_counts / (pos_counts + 1e-6))
    return weights


def train_test(data_model, optimizer, data, L_emb, edge_index, L_emb_edge,
               tr_mask, te_mask, epochs, Y):
    """返回每个epoch的指标"""
    model = data_model['model']
    epoch_aurocs = []
    epoch_auprcs = []

    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
        # 模型前向传播
        edge_index_train = dropout_adj(edge_index, p=0.3)[0]
        loss_inter, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
            data.x, edge_index_train, L_emb, L_emb_edge
        )

        class_weights = get_class_weights(Y[tr_mask])
        loss_G = F.binary_cross_entropy_with_logits(label_G[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_self = F.binary_cross_entropy_with_logits(label_self[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_neighbor = F.binary_cross_entropy_with_logits(label_neighbor[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_together = F.binary_cross_entropy_with_logits(label_together[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_concat = F.binary_cross_entropy_with_logits(label_concat[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_satment = F.binary_cross_entropy_with_logits(label_satment[tr_mask], Y[tr_mask], pos_weight=class_weights)
        loss_topk_fused = F.binary_cross_entropy_with_logits(final_output[tr_mask], Y[tr_mask], pos_weight=class_weights)

        loss_cls = loss_G + loss_self + loss_neighbor + loss_together + loss_concat + loss_satment + loss_topk_fused
        total_loss = loss_cls + data_model['lambdinter'] * loss_inter

        total_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
                data.x, edge_index, L_emb, L_emb_edge
            )

            pred = torch.sigmoid(final_output[te_mask]).cpu().numpy().ravel()
            precision, recall, _thresholds = metrics.precision_recall_curve(Y[te_mask].cpu().numpy(), pred)
            auc = metrics.roc_auc_score(Y[te_mask].cpu().numpy(), pred)
            auprc = metrics.auc(recall, precision)
            epoch_aurocs.append(auc)
            epoch_auprcs.append(auprc)
            print(f"Epoch {epoch+1}, Test AUC: {auc:.4f}, Test AUPRC: {auprc:.4f}")

    return epoch_aurocs, epoch_auprcs, auc, auprc


def trainPred_k_sets(input_dim, k_sets, data, L_emb, edge_index, L_emb_edge,
                     lr=0.001, epochs=200, lambdinter=0.005,
                     dropout=0.2, cancerType='pan-cancer', dataset='cpdb'):
    """收集每个epoch的指标（5折交叉验证）"""
    all_aurocs = np.zeros((epochs, 10, 5))
    all_auprcs = np.zeros((epochs, 10, 5))
    if cancerType == 'pan-cancer':
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        y_all = np.logical_or(data.y, data.y_te)
        mask_all = np.logical_or(data.mask, data.mask_te)
        print(mask_all.sum())
    else:
        label, label_pos, label_neg = load_label_single(cancerType)
        random.shuffle(label_pos)
        random.shuffle(label_neg)
        print(label.sum())
        y_train_pos = label_pos[:int(0.75 * len(label_pos))]
        y_test_pos = label_pos[int(0.75 * len(label_pos)):]
        y_train_neg = label_neg[:int(0.75 * len(label_neg))]
        y_test_neg = label_neg[int(0.75 * len(label_neg)):]
        l = len(label)
        l1 = int(len(y_train_pos) / 5)
        l2 = int(len(y_train_neg) / 5)
        Y = label

    list_aurocs = np.zeros((10, 5))
    list_auprcs = np.zeros((10, 5))

    n_exp = int(os.environ.get('N_EXP', 10))
    n_fold = int(os.environ.get('N_FOLD', 5))

    for exp_id in range(n_exp):
        for fold_id in range(n_fold):
            print(f"\nExp {exp_id+1}/{n_exp} | Fold {fold_id+1}/{n_fold}")
            
            if cancerType == 'pan-cancer':
                _, _, tr_mask, te_mask = k_sets[exp_id][fold_id]
                print(tr_mask.sum())
                print(te_mask.sum())
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)
            else:
                tr_mask, te_mask = sample_division_single(y_train_pos, y_train_neg, l, l1, l2, fold_id)
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)
                print(tr_mask.sum())
                print(te_mask.sum())
            
            model = combine_net_gate_without_ac(input_dim=input_dim, lambdinter=lambdinter, dropout=dropout).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            aurocs, auprcs, auc, auprc = train_test(
                data_model={
                    'model': model,
                    'lambdinter': lambdinter
                },
                optimizer=optimizer,
                data=data,
                L_emb=L_emb,
                edge_index=edge_index.to(device),
                L_emb_edge=L_emb_edge,
                tr_mask=train_mask.nonzero().squeeze(),
                te_mask=test_mask,
                epochs=epochs,
                Y=Y
            )
            
            all_aurocs[:, exp_id, fold_id] = aurocs
            all_auprcs[:, exp_id, fold_id] = auprcs
            list_aurocs[exp_id, fold_id] = auprc
            list_auprcs[exp_id, fold_id] = auc
            if cancerType == 'pan-cancer':
                np.savetxt(os.path.join(RESULT_DIR, 'pan-cancer_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(RESULT_DIR, 'pan-cancer_auprc.txt'), list_auprcs, fmt='%.6f')
            else:
                single_dir = os.path.join(RESULT_DIR, 'single')
                os.makedirs(single_dir, exist_ok=True)
                np.savetxt(os.path.join(single_dir, dataset + '_' + cancerType + '_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(single_dir, dataset + '_' + cancerType + '_auprc.txt'), list_auprcs, fmt='%.6f')

    save_results_to_file(list_aurocs, list_auprcs, cancerType, dataset=dataset, lr=lr, dropout=dropout, lambdinter=lambdinter)
    results = 0
    return results


def load_leakage_splits(audit_file_path=None):
    """
    从审计文件 src/Gemma_Vocabulary_Leakage_Audit.xlsx 加载 Clean 和 Hit 划分。
    
    规则：
    1. 使用 'Gene-level Flags' Sheet 中的 'LLM combined | Any exact label-like term' 列作为分组标记。
    2. 只保留有标签的 Driver 和 Non-driver 基因，排除 Unknown 基因。
    3. 按 Code_Index 与 CPDB 图节点严格对齐（0..13626）。
    """
    if audit_file_path is None:
        audit_file_path = os.path.join(BASE_DIR, 'src', 'Gemma_Vocabulary_Leakage_Audit.xlsx')

    if not os.path.exists(audit_file_path):
        raise FileNotFoundError(f"未找到审计文件: {audit_file_path}")

    print(f"正在读取审计文件: {audit_file_path} ...")
    df = pd.read_excel(audit_file_path, sheet_name='Gene-level Flags')

    assert len(df) == 13627, f"基因节点数不匹配: 期望 13627, 实际 {len(df)}"
    assert (df['Code_Index'].values == np.arange(len(df))).all(), "Code_Index 必须与节点索引 0..13626 严格对齐"

    flag_col = 'LLM combined | Any exact label-like term'
    if flag_col not in df.columns:
        raise KeyError(f"Sheet 'Gene-level Flags' 中未找到列 '{flag_col}'")

    labeled_mask = (df['Label_Status'] == 'Labeled').values
    hit_flag = (df[flag_col] == 1).values

    clean_mask = labeled_mask & (~hit_flag)
    hit_mask = labeled_mask & hit_flag

    return clean_mask, hit_mask, df


def trainPred_fixed_split(input_dim, train_mask, test_mask, data, L_emb, edge_index, L_emb_edge,
                          lr=0.0005, epochs=160, lambdinter=0.001, dropout=0.3,
                          split_name='clean_to_hit', n_exp=10, Y=None, base_seed=42):
    """
    针对标签泄露词审计的固定划分训练评估函数。
    不使用 k_sets.pkl 划分，而是使用由审计文件确定的固定 train_mask 和 test_mask。
    保留原有 CPDB 数据、网络、LLM embedding、模型结构与超参数。
    """
    tr_indices = train_mask.nonzero().squeeze()
    te_indices = test_mask

    train_drivers = (Y[train_mask] == 1).sum().item()
    train_nondrivers = (Y[train_mask] == 0).sum().item()
    test_drivers = (Y[test_mask] == 1).sum().item()
    test_nondrivers = (Y[test_mask] == 0).sum().item()

    print(f"\n{'='*60}")
    print(f"Running Leakage Split Experiment: [{split_name}]")
    print(f"  Training set : {train_mask.sum().item()} genes (Drivers: {int(train_drivers)}, Non-drivers: {int(train_nondrivers)})")
    print(f"  Testing set  : {test_mask.sum().item()} genes (Drivers: {int(test_drivers)}, Non-drivers: {int(test_nondrivers)})")
    print(f"  Experiments  : {n_exp} independent runs, {epochs} epochs each")
    print(f"  Hyperparameters: lr={lr}, dropout={dropout}, lambdinter={lambdinter}")
    print(f"{'='*60}\n")

    all_aurocs = np.zeros((epochs, n_exp))
    all_auprcs = np.zeros((epochs, n_exp))
    final_aurocs = np.zeros(n_exp)
    final_auprcs = np.zeros(n_exp)
    best_aurocs = np.zeros(n_exp)
    best_auprcs = np.zeros(n_exp)

    for exp_id in range(n_exp):
        seed = base_seed + exp_id
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n--- [Split: {split_name}] Experiment {exp_id + 1}/{n_exp} (Seed: {seed}) ---")

        model = combine_net_gate_without_ac(input_dim=input_dim, lambdinter=lambdinter, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        aurocs, auprcs, auc, auprc = train_test(
            data_model={
                'model': model,
                'lambdinter': lambdinter
            },
            optimizer=optimizer,
            data=data,
            L_emb=L_emb,
            edge_index=edge_index.to(device),
            L_emb_edge=L_emb_edge,
            tr_mask=tr_indices,
            te_mask=te_indices,
            epochs=epochs,
            Y=Y
        )

        all_aurocs[:, exp_id] = aurocs
        all_auprcs[:, exp_id] = auprcs
        final_aurocs[exp_id] = auc
        final_auprcs[exp_id] = auprc

        best_epoch_idx = np.argmax(auprcs)
        best_aurocs[exp_id] = aurocs[best_epoch_idx]
        best_auprcs[exp_id] = auprcs[best_epoch_idx]

    mean_auc, std_auc = final_aurocs.mean(), final_aurocs.std()
    mean_auprc, std_auprc = final_auprcs.mean(), final_auprcs.std()
    mean_best_auc, std_best_auc = best_aurocs.mean(), best_aurocs.std()
    mean_best_auprc, std_best_auprc = best_auprcs.mean(), best_auprcs.std()

    print(f"\n{'='*60}")
    print(f"Summary Results for Leakage Split: [{split_name}]")
    print(f"Final Epoch  -> AUROC: {mean_auc:.4f} ± {std_auc:.4f} | AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
    print(f"Best  Epoch  -> AUROC: {mean_best_auc:.4f} ± {std_best_auc:.4f} | AUPRC: {mean_best_auprc:.4f} ± {std_best_auprc:.4f}")
    print(f"{'='*60}\n")

    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    summary_path = os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_summary.txt")
    np.savetxt(os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_auroc.txt"), final_aurocs, fmt='%.6f')
    np.savetxt(os.path.join(res_dir, f"pan-cancer_leakage_{split_name}_auprc.txt"), final_auprcs, fmt='%.6f')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"Leakage Split Experiment: {split_name}\n")
        f.write(f"Training set: {train_mask.sum().item()} (Drivers: {int(train_drivers)}, Non-drivers: {int(train_nondrivers)})\n")
        f.write(f"Testing set:  {test_mask.sum().item()} (Drivers: {int(test_drivers)}, Non-drivers: {int(test_nondrivers)})\n")
        f.write(f"Experiments: {n_exp}, Epochs per exp: {epochs}\n")
        f.write(f"Hyperparameters: lr={lr}, dropout={dropout}, lambdinter={lambdinter}\n")
        f.write('-' * 40 + '\n')
        f.write(f"Final Epoch Metric:\n")
        f.write(f"  AUROC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"  AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}\n")
        f.write(f"  All AUROC per exp: {np.array2string(final_aurocs, precision=4)}\n")
        f.write(f"  All AUPRC per exp: {np.array2string(final_auprcs, precision=4)}\n")
        f.write('-' * 40 + '\n')
        f.write(f"Best Epoch Metric:\n")
        f.write(f"  AUROC: {mean_best_auc:.4f} ± {std_best_auc:.4f}\n")
        f.write(f"  AUPRC: {mean_best_auprc:.4f} ± {std_best_auprc:.4f}\n")
        f.write(f"  All Best AUROC per exp: {np.array2string(best_aurocs, precision=4)}\n")
        f.write(f"  All Best AUPRC per exp: {np.array2string(best_auprcs, precision=4)}\n")

    return final_aurocs, final_auprcs


def main():
    parser = argparse.ArgumentParser(description="BIMODriver Training and Evaluation")
    parser.add_argument('--split', type=str, default=os.environ.get('SPLIT', 'cv'),
                        help="划分模式: 'cv' (默认5折交叉验证), 'clean_to_hit' (Clean训练->Hit测试), 'hit_to_clean' (Hit训练->Clean测试)")
    parser.add_argument('--dataset', type=str, default='cpdb', choices=['cpdb', 'string'],
                        help="数据集类型 ('cpdb' 或 'string')")
    parser.add_argument('--cancerType', type=str, default='pan-cancer',
                        help="癌种名称 ('pan-cancer' 或特定单癌种)")
    parser.add_argument('--epochs', type=int, default=int(os.environ.get('EPOCHS', 160)),
                        help="训练轮数 (默认 160)")
    parser.add_argument('--n_exp', type=int, default=int(os.environ.get('N_EXP', 10)),
                        help="独立实验次数 (默认 10)")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="学习率 (默认 0.0005)")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout 率 (默认 0.3)")
    parser.add_argument('--lambdinter', type=float, default=0.001,
                        help="特征对齐损失权重 (默认 0.001)")
    parser.add_argument('--audit_file', type=str, default=None,
                        help="审计文件路径 (默认 src/Gemma_Vocabulary_Leakage_Audit.xlsx)")
    parser.add_argument('--base_seed', type=int, default=42,
                        help="随机种子基准值 (默认 42)")

    args = parser.parse_args()

    split_mode = args.split.lower().replace('->', '_to_').replace('-', '_')

    # 加载 CPDB 数据
    if args.dataset == 'cpdb':
        data = torch.load(os.path.join(DATA_DIR, "CPDB", "CPDB_new_data.pt"))
        data = data.to(device)
        data.x = data.x[:, :48]
        if args.cancerType != 'pan-cancer':
            cancerType_dict = {
                'kirc': [0, 16, 32], 'brca': [1, 17, 33], 'prad': [3, 19, 35],
                'stad': [4, 20, 36], 'hnsc': [5, 21, 37], 'luad': [6, 22, 38],
                'thca': [7, 23, 39], 'blca': [8, 24, 40], 'esca': [9, 25, 41],
                'lihc': [10, 26, 42], 'ucec': [11, 27, 43], 'coad': [12, 28, 44],
                'lusc': [13, 29, 45], 'cesc': [14, 30, 46], 'kirp': [15, 31, 47]
            }
            data.x = data.x[:, cancerType_dict[args.cancerType]]

        datas = torch.load(os.path.join(DATA_DIR, "CPDB", "Str_feature.pkl")).to(device)
        data.x = torch.cat((data.x, datas), 1)
        data = data.to(device)

        statement = torch.load(os.path.join(DATA_DIR, "CPDB", "PAN-CANCER_statement_features.pt")).to(device)
        L_emb = {
            'self_emb': statement[:, 0:768],
            'neighbor_emb': statement[:, 768:1536],
            'together_emb': statement[:, 1536:2304]
        }
        L_emb_edge = torch.load(os.path.join(DATA_DIR, "cpdb_network_LLM", "merged_k5_edge_index.pt")).to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)

        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)

    elif args.dataset == 'string':
        if split_mode in ['clean_to_hit', 'hit_to_clean']:
            raise ValueError("标签泄露审计划分目前仅支持 CPDB 数据集")
        data = torch.load(os.path.join(DATA_DIR, "STRING", "STRING_data.pkl"))
        data = data.to(device)
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        data.x = data.x[:, :48]

        datas = torch.load(os.path.join(DATA_DIR, "STRING", "Str_feature.pkl")).to(device)
        data.x = torch.cat((data.x, datas), 1)
        data = data.to(device)

        with open(os.path.join(DATA_DIR, "STRING", "k_sets.pkl"), 'rb') as handle:
            k_sets = torch.load(os.path.join(DATA_DIR, "STRING", "k_sets.pkl"))

        statement = torch.load(os.path.join(DATA_DIR, "STRING", "PAN-CANCER_string_new_neiber2.pt")).to(device)
        L_emb = {
            'self_emb': statement[:, 0:768],
            'neighbor_emb': statement[:, 768:1536],
            'together_emb': statement[:, 1536:2304]
        }
        L_emb_edge = torch.load(os.path.join(DATA_DIR, "string_network_LLM", "merged_k5_edge_index.pt")).to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)
    else:
        raise ValueError("Unsupported dataset. Please choose 'cpdb' or 'string'.")

    input_dim = data.x.shape[1]
    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)

    # 分支 1：标签泄露词划分实验
    if split_mode in ['clean_to_hit', 'hit_to_clean']:
        clean_mask, hit_mask, audit_df = load_leakage_splits(args.audit_file)

        if split_mode == 'clean_to_hit':
            train_mask = torch.tensor(clean_mask).bool().to(device)
            test_mask = torch.tensor(hit_mask).bool().to(device)
            split_display_name = 'Clean -> Hit'
        else:
            train_mask = torch.tensor(hit_mask).bool().to(device)
            test_mask = torch.tensor(clean_mask).bool().to(device)
            split_display_name = 'Hit -> Clean'

        trainPred_fixed_split(
            input_dim=input_dim,
            train_mask=train_mask,
            test_mask=test_mask,
            data=data,
            L_emb=L_emb,
            edge_index=pb,
            L_emb_edge=L_emb_edge,
            lr=args.lr,
            epochs=args.epochs,
            lambdinter=args.lambdinter,
            dropout=args.dropout,
            split_name=split_mode,
            n_exp=args.n_exp,
            Y=Y,
            base_seed=args.base_seed
        )

    # 分支 2：原始 5 折交叉验证实验
    elif split_mode == 'cv':
        with open(os.path.join(DATA_DIR, "CPDB", "k_sets.pkl"), 'rb') as handle:
            k_sets = pickle.load(handle)

        print(f"\nRunning original 5-fold CV for {args.cancerType} on {args.dataset}...")
        trainPred_k_sets(
            input_dim=input_dim,
            k_sets=k_sets,
            data=data,
            L_emb=L_emb,
            edge_index=pb,
            L_emb_edge=L_emb_edge,
            lr=args.lr,
            epochs=args.epochs,
            lambdinter=args.lambdinter,
            dropout=args.dropout,
            cancerType=args.cancerType,
            dataset=args.dataset
        )
    else:
        raise ValueError(f"未知的划分模式: {args.split}。可选模式: 'cv', 'clean_to_hit', 'hit_to_clean'")


if __name__ == '__main__':
    main()
