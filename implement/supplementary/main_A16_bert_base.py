# -*- coding: utf-8 -*-
"""Table A16: BIMODriver with BERT-base-cased semantic embeddings."""
import numpy as np
import pandas as pd
import time
import pickle
import random
import warnings
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / 'data'
if str(REPO_ROOT / 'BIMODriver') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'BIMODriver'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
import gcnPreprocessing
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops, add_self_loops
import copy
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split as sk_train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn import linear_model
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from model_A16_bert_base import *


RESULT_ROOT = str(REPO_ROOT / 'result' / 'A16')


def _append_csv_row(path, fieldnames, row):
    """立即追加保存一行结果，避免程序中断后已完成结果丢失。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def create_run_result_dir(cancerType, dataset, lr, dropout, lambdinter, epochs):
    """为每次程序运行创建独立结果目录，避免覆盖历史结果。"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = (
        f'{dataset}_{cancerType}_{timestamp}'
        f'_lr{lr}_dropout{dropout}_lambda{lambdinter}'
    )
    run_dir = os.path.join(RESULT_ROOT, 'runs', run_name)
    os.makedirs(run_dir, exist_ok=False)

    metadata_path = os.path.join(run_dir, 'run_info.txt')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write(f'start_time={datetime.now().isoformat()}\n')
        f.write(f'dataset={dataset}\n')
        f.write(f'cancerType={cancerType}\n')
        f.write(f'lr={lr}\n')
        f.write(f'dropout={dropout}\n')
        f.write(f'lambdinter={lambdinter}\n')
        f.write(f'epochs={epochs}\n')
        f.write('experiments=10\n')
        f.write('folds_per_experiment=5\n')

    print(f'本次运行的全部详细结果将保存到: {run_dir}')
    return run_dir


# ================== 修改：save_epoch_result 增加 epoch_time 参数 ==================
def save_epoch_result(run_dir, exp_id, fold_id, epoch, auc, auprc, total_loss, epoch_time):
    """保存每一个 epoch 的结果，包含 epoch 耗时（秒）。"""
    path = os.path.join(run_dir, 'epoch_results.csv')
    _append_csv_row(
        path,
        ['exp', 'fold', 'epoch', 'auc', 'auprc', 'total_loss', 'epoch_time_seconds'],
        {
            'exp': exp_id + 1,
            'fold': fold_id + 1,
            'epoch': epoch + 1,
            'auc': f'{auc:.10f}',
            'auprc': f'{auprc:.10f}',
            'total_loss': f'{float(total_loss):.10f}',
            'epoch_time_seconds': f'{epoch_time:.4f}',
        }
    )


# ================== 修改：save_fold_result 增加 fold_time 参数 ==================
def save_fold_result(run_dir, exp_id, fold_id, aurocs, auprcs, fold_time):
    """每完成一个 Exp/Fold，立即保存最终值和最佳值，同时记录该 Fold 耗时。"""
    aurocs = np.asarray(aurocs, dtype=float)
    auprcs = np.asarray(auprcs, dtype=float)
    best_auc_idx = int(np.argmax(aurocs))
    best_auprc_idx = int(np.argmax(auprcs))
    path = os.path.join(run_dir, 'fold_results.csv')
    _append_csv_row(
        path,
        [
            'exp', 'fold', 'final_auc', 'final_auprc',
            'best_auc', 'best_auc_epoch', 'best_auprc', 'best_auprc_epoch',
            'fold_time_seconds'                # 新增列
        ],
        {
            'exp': exp_id + 1,
            'fold': fold_id + 1,
            'final_auc': f'{aurocs[-1]:.10f}',
            'final_auprc': f'{auprcs[-1]:.10f}',
            'best_auc': f'{aurocs[best_auc_idx]:.10f}',
            'best_auc_epoch': best_auc_idx + 1,
            'best_auprc': f'{auprcs[best_auprc_idx]:.10f}',
            'best_auprc_epoch': best_auprc_idx + 1,
            'fold_time_seconds': f'{fold_time:.2f}',
        }
    )


def save_progress_snapshot(run_dir, all_aurocs, all_auprcs, list_aurocs, list_auprcs):
    """保存当前运行进度快照；只覆盖本次运行目录内的快照。"""
    np.savez_compressed(
        os.path.join(run_dir, 'progress_snapshot.npz'),
        all_aurocs=all_aurocs,
        all_auprcs=all_auprcs,
        legacy_list_aurocs=list_aurocs,
        legacy_list_auprcs=list_auprcs,
    )


def save_results_to_file(auroc, auprc, cancerType, dataset='cpdb', lr=0.001, dropout=0.2, lambdinter=0.005):
    if cancerType == 'pan-cancer':
        if dataset == 'cpdb':
            path = os.path.join(RESULT_ROOT, 'pan-cancer.txt')
        elif dataset == 'string':
            path = os.path.join(RESULT_ROOT, 'pan-cancer_string.txt')
        else:
            raise ValueError("Unsupported dataset for pan-cancer results.")
    else:
        path = os.path.join(RESULT_ROOT, 'single', 'single.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write('--' * 20 + '\n')
        f.write(f"Dropout Rate: {dropout}, Learning Rate: {lr}, Lambda Inter: {lambdinter}\n")
        f.write(f"Results for {cancerType}:\n")
        f.write(f"AUPR: {auroc.mean():.4f} +/- {auroc.std():.4f}\n")
        f.write(str(auroc))
        f.write("\n")
        f.write(f"AUC: {auprc.mean():.4f} +/- {auprc.std():.4f}\n")
        f.write(str(auprc))
        f.write("\n")


def load_label_single(cancerType):
    path = str(DATA_DIR / "CPDB" / "Specific cancer") + "/"
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
    # neg_counts = 2983 - pos_counts
    neg_counts = labels.shape[0] - pos_counts
    weights = (neg_counts / (pos_counts + 1e-6))
    return weights


# ================== 修改：train_test 内部对每个 epoch 计时 ==================
def train_test(data_model, optimizer, data, L_emb, edge_index, L_emb_edge,
               tr_mask, te_mask, epochs, Y, run_dir=None, exp_id=None, fold_id=None):
    """返回每个epoch的指标，并记录每个epoch的耗时"""
    model = data_model['model']
    epoch_aurocs = []
    epoch_auprcs = []

    for epoch in range(epochs):
        epoch_start = time.time()   # 记录 epoch 开始时间

        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
        # 模型前向传播
        edge_index_train = dropout_adj(edge_index, p=0.3)[0]
        loss_inter, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
            data.x, edge_index_train, L_emb, L_emb_edge)

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

        # ===== 评估阶段 =====
        model.eval()
        with torch.no_grad():
            _, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
                data.x, edge_index, L_emb, L_emb_edge)

            pred = torch.sigmoid(final_output[te_mask]).cpu().numpy().ravel()
            precision, recall, _thresholds = metrics.precision_recall_curve(Y[te_mask].cpu().numpy(), pred)
            auc = metrics.roc_auc_score(Y[te_mask].cpu().numpy(), pred)
            auprc = metrics.auc(recall, precision)
            epoch_aurocs.append(auc)
            epoch_auprcs.append(auprc)

            epoch_elapsed = time.time() - epoch_start   # 计算该 epoch 耗时

            print(f"Epoch {epoch+1}, Test AUC: {auc:.4f}, Test AUPRC: {auprc:.4f}, Time: {epoch_elapsed:.2f}s")

            if run_dir is not None:
                save_epoch_result(run_dir, exp_id, fold_id, epoch, auc, auprc,
                                  total_loss.detach().cpu().item(), epoch_elapsed)

    return epoch_aurocs, epoch_auprcs, auc, auprc


# ================== 修改：trainPred_k_sets 增加总时间记录和 fold 计时 ==================
def trainPred_k_sets(input_dim, k_sets, data, L_emb, edge_index, L_emb_edge,
                     lr=0.001, epochs=200, lambdinter=0.005,
                     dropout=0.2, cancerType='pan-cancer', dataset='cpdb'):
    """收集所有epoch的指标，并记录整体运行时间及每个fold的耗时"""
    
    # 记录整个函数运行的开始时间
    total_start_time = time.time()

    # 初始化存储结构 [epoch][experiment][fold]
    all_aurocs = np.zeros((epochs, 10, 5))
    all_auprcs = np.zeros((epochs, 10, 5))
    run_dir = create_run_result_dir(cancerType, dataset, lr, dropout, lambdinter, epochs)

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

    # 遍历10次独立实验
    for exp_id in range(10):
        for fold_id in range(5):
            # 记录每个 fold 的开始时间
            fold_start_time = time.time()

            print(f"\nExp {exp_id+1}/10 | Fold {fold_id+1}/5")
            
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
            
            # 初始化模型
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
                Y=Y,
                run_dir=run_dir,
                exp_id=exp_id,
                fold_id=fold_id
            )
            
            # 存储结果
            all_aurocs[:, exp_id, fold_id] = aurocs
            all_auprcs[:, exp_id, fold_id] = auprcs
            list_aurocs[exp_id, fold_id] = auprc
            list_auprcs[exp_id, fold_id] = auc

            # 计算该 fold 耗时
            fold_elapsed = time.time() - fold_start_time

            # 保存 fold 结果（含耗时）
            save_fold_result(run_dir, exp_id, fold_id, aurocs, auprcs, fold_elapsed)
            save_progress_snapshot(run_dir, all_aurocs, all_auprcs, list_aurocs, list_auprcs)

            if cancerType == 'pan-cancer':
                np.savetxt(os.path.join(RESULT_ROOT, 'pan-cancer_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(RESULT_ROOT, 'pan-cancer_auprc.txt'), list_auprcs, fmt='%.6f')
            else:
                single_result_dir = os.path.join(RESULT_ROOT, 'single')
                os.makedirs(single_result_dir, exist_ok=True)
                np.savetxt(os.path.join(single_result_dir, dataset + '_' + cancerType + '_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(single_result_dir, dataset + '_' + cancerType + '_auprc.txt'), list_auprcs, fmt='%.6f')

    save_results_to_file(list_aurocs, list_auprcs, cancerType, dataset=dataset, lr=lr, dropout=dropout, lambdinter=lambdinter)

    save_best_epoch_results(
        list_aurocs, list_auprcs,
        all_aurocs, all_auprcs,
        cancerType=cancerType, dataset=dataset,
        lr=lr, dropout=dropout, lambdinter=lambdinter,
        epochs=epochs
    )

    # ===== 记录总运行时间并写入 run_info.txt =====
    total_elapsed = time.time() - total_start_time
    with open(os.path.join(run_dir, 'run_info.txt'), 'a', encoding='utf-8') as f:
        f.write(f'total_elapsed_seconds={total_elapsed:.2f}\n')
        f.write(f'total_elapsed_hms={time.strftime("%H:%M:%S", time.gmtime(total_elapsed))}\n')

    with open(os.path.join(run_dir, 'RUN_COMPLETED.txt'), 'w', encoding='utf-8') as f:
        f.write(f'completed_time={datetime.now().isoformat()}\n')
        f.write(f'total_elapsed_seconds={total_elapsed:.2f}\n')
        f.write(f'total_elapsed_hms={time.strftime("%H:%M:%S", time.gmtime(total_elapsed))}\n')

    print(f'本次运行已完成，总耗时 {total_elapsed:.2f} 秒 ({time.strftime("%H:%M:%S", time.gmtime(total_elapsed))})')
    print(f'详细结果保存在: {run_dir}')
    results = 0
    return results


# ===== 主程序 =====
cancers = ['pan-cancer']
dataset = 'cpdb'  # 'cpdb' or 'string'
for cancerType in cancers:
    if dataset == 'cpdb':
        data = torch.load(DATA_DIR / "CPDB" / "CPDB_new_data.pt")
        data = data.to(device)
        data.x = data.x[:, :48]
        if cancerType == 'pan-cancer':
            data.x = data.x[:, :48]
        else:
            cancerType_dict = {
                'kirc': [0, 16, 32],
                'brca': [1, 17, 33],
                'prad': [3, 19, 35],
                'stad': [4, 20, 36],
                'hnsc': [5, 21, 37],
                'luad': [6, 22, 38],
                'thca': [7, 23, 39],
                'blca': [8, 24, 40],
                'esca': [9, 25, 41],
                'lihc': [10, 26, 42],
                'ucec': [11, 27, 43],
                'coad': [12, 28, 44],
                'lusc': [13, 29, 45],
                'cesc': [14, 30, 46],
                'kirp': [15, 31, 47]
            }
            data.x = data.x[:, cancerType_dict[cancerType]]

        datas = torch.load(DATA_DIR / "CPDB" / "Str_feature.pkl")
        data.x = torch.cat((data.x, datas), 1)
        data = data.to(device)

        with open(DATA_DIR / "CPDB" / "k_sets.pkl", 'rb') as handle:
            k_sets = pickle.load(handle)

        statement = torch.load(DATA_DIR / "CPDB" / "PAN-CANCER_statement_features.pt").to(device)
        L_emb = {}
        L_emb['self_emb'] = statement[:, 0:768]
        L_emb['neighbor_emb'] = statement[:, 768:1536]
        L_emb['together_emb'] = statement[:, 1536:2304]

        L_emb_edge = torch.load(DATA_DIR / "cpdb_network_LLM" / "merged_k5_edge_index.pt").to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)

    elif dataset == 'string':
        data = torch.load(DATA_DIR / "STRING" / "STRING_data.pkl")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        y_all = np.logical_or(data.y, data.y_te)
        mask_all = np.logical_or(data.mask, data.mask_te)
        data.x = data.x[:, :48]

        datas = torch.load(DATA_DIR / "STRING" / "Str_feature.pkl").to(device)
        data.x = torch.cat((data.x, datas), 1)

        data = data.to(device)
        k_sets = torch.load(DATA_DIR / "STRING" / "k_sets.pkl")

        statement = torch.load(DATA_DIR / "STRING" / "PAN-CANCER_string_new_neiber2.pt").to(device)
        L_emb = {}
        L_emb['self_emb'] = statement[:, 0:768]
        L_emb['neighbor_emb'] = statement[:, 768:1536]
        L_emb['together_emb'] = statement[:, 1536:2304]
        L_emb_edge = torch.load(DATA_DIR / "string_network_LLM" / "merged_k5_edge_index.pt").to(device)

        if isinstance(L_emb, dict):
            for key in L_emb:
                if torch.is_tensor(L_emb[key]):
                    L_emb[key] = L_emb[key].to(device)
    else:
        raise ValueError("Unsupported dataset. Please choose 'cpdb' or 'string'.")

    input_dim = data.x.shape[1]
    pb, _ = remove_self_loops(data.edge_index)
    pb, _ = add_self_loops(pb)
    E = data.edge_index
    EPOCH = 160

    dropout_rates = [0.3]
    lrs = [0.0005]
    lambdinters = [0.001]

    for dropoutrate in dropout_rates:
        for lr in lrs:
            for lambdinter in lambdinters:
                print(f"\nTraining for cancer type: {cancerType}, dropout rate: {dropoutrate}, learning rate: {lr}, lambda inter: {lambdinter}")

                results = trainPred_k_sets(
                    input_dim=input_dim,
                    k_sets=k_sets,
                    data=data,
                    L_emb=L_emb,
                    edge_index=pb,
                    L_emb_edge=L_emb_edge,
                    lr=lr,
                    epochs=EPOCH,
                    lambdinter=lambdinter,
                    dropout=dropoutrate,
                    cancerType=cancerType,
                    dataset=dataset
                )
