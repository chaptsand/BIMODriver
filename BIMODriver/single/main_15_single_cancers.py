# -*- coding: utf-8 -*-
import os
import sys
import csv
import time
import yaml
import pickle
import random
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj, remove_self_loops, add_self_loops

# 模块与路径解析
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIMODRIVER_DIR = os.path.dirname(SCRIPT_DIR)
BASE_DIR = os.path.dirname(BIMODRIVER_DIR)

if BIMODRIVER_DIR not in sys.path:
    sys.path.insert(0, BIMODRIVER_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model import combine_net_gate_without_ac, save_best_epoch_results

device = torch.device('cuda' if torch.cuda.is_available() and os.environ.get('DEVICE') != 'cpu' else 'cpu')
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULT_ROOT = os.environ.get('BIMODRIVER_RESULT_DIR', os.path.join(BASE_DIR, 'result'))
CONFIG_PATH = os.environ.get(
    'BIMODRIVER_MOE_CONFIG',
    os.path.join(SCRIPT_DIR, 'MOE-config-15cancer.yaml')
)

with open(CONFIG_PATH, 'r', encoding='utf-8') as config_file:
    MOE_CONFIG = yaml.safe_load(config_file)

SPLIT_SEED = int(MOE_CONFIG['COMMON']['seed'])
torch.manual_seed(SPLIT_SEED)
np.random.seed(SPLIT_SEED)
random.seed(SPLIT_SEED)


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


def create_run_result_dir(cancerType, dataset, lr, dropout, lambdinter, epochs, n_exp=10, n_fold=5):
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
        f.write(f'experiments={n_exp}\n')
        f.write(f'folds_per_experiment={n_fold}\n')

    print(f'本次运行的全部详细结果将保存到: {run_dir}')
    return run_dir


def save_epoch_result(run_dir, exp_id, fold_id, epoch, auc, auprc, total_loss, epoch_time):
    """保存每一个 epoch 的结果，包含 epoch 耗时（秒）。"""
    path = os.path.join(run_dir, 'epoch_results.csv')
    _append_csv_row(
        path,
        ['exp', 'fold', 'epoch', 'auroc', 'auprc', 'total_loss', 'epoch_time_seconds'],
        {
            'exp': exp_id + 1,
            'fold': fold_id + 1,
            'epoch': epoch + 1,
            'auroc': f'{auc:.10f}',
            'auprc': f'{auprc:.10f}',
            'total_loss': f'{float(total_loss):.10f}',
            'epoch_time_seconds': f'{epoch_time:.4f}',
        }
    )


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
            'exp', 'fold', 'final_auroc', 'final_auprc',
            'best_auroc', 'best_auroc_epoch', 'best_auprc', 'best_auprc_epoch',
            'fold_time_seconds'
        ],
        {
            'exp': exp_id + 1,
            'fold': fold_id + 1,
            'final_auroc': f'{aurocs[-1]:.10f}',
            'final_auprc': f'{auprcs[-1]:.10f}',
            'best_auroc': f'{aurocs[best_auc_idx]:.10f}',
            'best_auroc_epoch': best_auc_idx + 1,
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
    """汇总并保存最终评测指标到文本文件。"""
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
        f.write(f"AUROC: {auroc.mean():.4f} ± {auroc.std():.4f}\n")
        f.write(str(auroc) + "\n")
        f.write(f"AUPRC: {auprc.mean():.4f} ± {auprc.std():.4f}\n")
        f.write(str(auprc) + "\n")


def load_label_single(cancerType):
    """加载单癌种正负标签，并对标签索引执行严格边界检查。"""
    path = os.path.join(DATA_DIR, "CPDB", "Specific cancer") + "/"
    label = np.loadtxt(path + "label_file-P-" + cancerType + ".txt")
    Y = torch.tensor(label).type(torch.FloatTensor).to(device).unsqueeze(1)
    label_pos = np.atleast_1d(
        np.loadtxt(path + "pos-" + cancerType + ".txt", dtype=int)
    )
    label_neg = np.atleast_1d(
        np.loadtxt(path + "neg.txt", dtype=int)
    )
    max_index = len(label) - 1
    label_pos = [int(i) for i in label_pos if 0 <= int(i) <= max_index]
    label_neg = [int(i) for i in label_neg if 0 <= int(i) <= max_index]
    return Y, label_pos, label_neg


def sample_division_single(pos_label, neg_label, l, l1, l2, i):
    """平衡划分单癌种 5 折交叉验证训练集与测试集掩码。"""
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
               tr_mask, te_mask, epochs, Y, run_dir=None, exp_id=None, fold_id=None):
    """返回每个 epoch 的指标，并记录每个 epoch 的耗时。"""
    model = data_model['model']
    epoch_aurocs = []
    epoch_auprcs = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # ===== 训练阶段 =====
        model.train()
        optimizer.zero_grad()
        
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

        # ===== 评估阶段 =====
        model.eval()
        with torch.no_grad():
            _, label_G, label_self, label_neighbor, label_together, label_concat, label_satment, final_output = model(
                data.x, edge_index, L_emb, L_emb_edge
            )

            pred = torch.sigmoid(final_output[te_mask]).cpu().numpy().ravel()
            precision, recall, _thresholds = metrics.precision_recall_curve(Y[te_mask].cpu().numpy(), pred)
            auroc = metrics.roc_auc_score(Y[te_mask].cpu().numpy(), pred)
            auprc = metrics.auc(recall, precision)
            epoch_aurocs.append(auroc)
            epoch_auprcs.append(auprc)

            epoch_elapsed = time.time() - epoch_start

            if (epoch + 1) % 40 == 0 or (epoch + 1) == epochs:
                print(f"  Epoch {epoch+1:3d}/{epochs} | AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Elapsed: {epoch_elapsed:.2f}s")

            if run_dir is not None:
                save_epoch_result(run_dir, exp_id, fold_id, epoch, auroc, auprc,
                                  total_loss.detach().cpu().item(), epoch_elapsed)

    return epoch_aurocs, epoch_auprcs, auroc, auprc


def trainPred_k_sets(input_dim, k_sets, data, L_emb, edge_index, L_emb_edge,
                     lr=0.001, epochs=200, lambdinter=0.005,
                     dropout=0.2, cancerType='pan-cancer', dataset='cpdb',
                     n_exp=10, n_fold=5, top_k=None, split_file=None):
    """收集所有 epoch 的指标，并记录整体运行时间及每个 fold 的耗时。"""
    total_start_time = time.time()

    if top_k is None:
        top_k = 5 if cancerType == 'pan-cancer' else 4

    all_aurocs = np.zeros((epochs, n_exp, n_fold))
    all_auprcs = np.zeros((epochs, n_exp, n_fold))
    run_dir = create_run_result_dir(cancerType, dataset, lr, dropout, lambdinter, epochs, n_exp=n_exp, n_fold=n_fold)

    if cancerType == 'pan-cancer':
        Y = torch.tensor(np.logical_or(data.y, data.y_te)).type(torch.FloatTensor).to(device)
        y_all = np.logical_or(data.y, data.y_te)
        mask_all = np.logical_or(data.mask, data.mask_te)
        print(f"Pan-cancer total labeled genes: {mask_all.sum().item()}")
    else:
        Y, label_pos, label_neg = load_label_single(cancerType)
        target_split_path = split_file
        if target_split_path is None:
            candidate_path = os.path.join(DATA_DIR, "single_splits", f"{dataset}_{cancerType}_5fold_data_split.pkl")
            legacy_path = os.path.join(DATA_DIR, "data_splits", f"{dataset}_{cancerType}_5fold_data_split.pkl")
            if os.path.exists(candidate_path):
                target_split_path = candidate_path
            elif os.path.exists(legacy_path):
                target_split_path = legacy_path

        fixed_folds = []
        if target_split_path and os.path.exists(target_split_path):
            with open(target_split_path, 'rb') as f:
                saved_splits = pickle.load(f)
            for fold_id in range(5):
                f_info = saved_splits['folds'][fold_id]
                tr_m = torch.from_numpy(f_info['train_mask']).bool()
                te_m = torch.from_numpy(f_info['validation_mask']).bool()
                fixed_folds.append((tr_m, te_m))
            print(f"Cancer {cancerType} | [Info] Loaded precomputed 5-fold split from: {target_split_path}")
        else:
            split_rng = random.Random(SPLIT_SEED)
            shuffled_pos = list(label_pos)
            shuffled_neg = list(label_neg)
            split_rng.shuffle(shuffled_pos)
            split_rng.shuffle(shuffled_neg)
            print(f"Cancer {cancerType} | Positive: {len(shuffled_pos)}, Negative: {len(shuffled_neg)}, Labeled: {len(shuffled_pos) + len(shuffled_neg)}")
            l = len(Y)
            l1 = len(shuffled_pos) // 5
            l2 = len(shuffled_neg) // 5

            saved_splits = {
                'cancer_type': cancerType,
                'dataset': dataset,
                'split_seed': SPLIT_SEED,
                'all_positive_indices': np.asarray(shuffled_pos),
                'all_negative_indices': np.asarray(shuffled_neg),
                'folds': []
            }

            for fold_id in range(5):
                tr_mask, te_mask = sample_division_single(
                    shuffled_pos, shuffled_neg, l, l1, l2, fold_id
                )
                validation_pos = shuffled_pos[fold_id * l1:(fold_id + 1) * l1]
                validation_neg = shuffled_neg[fold_id * l2:(fold_id + 1) * l2]
                training_pos = list(set(shuffled_pos) - set(validation_pos))
                training_neg = list(set(shuffled_neg) - set(validation_neg))
                fixed_folds.append((tr_mask, te_mask))
                saved_splits['folds'].append({
                    'fold_id': fold_id + 1,
                    'train_positive_indices': np.asarray(training_pos),
                    'train_negative_indices': np.asarray(training_neg),
                    'validation_positive_indices': np.asarray(validation_pos),
                    'validation_negative_indices': np.asarray(validation_neg),
                    'train_indices': tr_mask.nonzero(as_tuple=True)[0].cpu().numpy(),
                    'validation_indices': te_mask.nonzero(as_tuple=True)[0].cpu().numpy(),
                    'train_mask': tr_mask.cpu().numpy(),
                    'validation_mask': te_mask.cpu().numpy()
                })

            split_path = os.path.join(run_dir, f'{cancerType}_5fold_data_split.pkl')
            with open(split_path, 'wb') as f:
                pickle.dump(saved_splits, f)
            print(f'{cancerType} 的固定五折划分已保存到: {split_path}')

            # 同时将生成的标准划分固化到 data/single_splits/ 供后续运行快速复用
            dataset_split_dir = os.path.join(DATA_DIR, "single_splits")
            os.makedirs(dataset_split_dir, exist_ok=True)
            dataset_split_path = os.path.join(dataset_split_dir, f"{dataset}_{cancerType}_5fold_data_split.pkl")
            if not os.path.exists(dataset_split_path):
                with open(dataset_split_path, 'wb') as f:
                    pickle.dump(saved_splits, f)
                print(f"Cancer {cancerType} | [Info] Generated and saved 5-fold split to: {dataset_split_path}")

    list_aurocs = np.zeros((n_exp, n_fold))
    list_auprcs = np.zeros((n_exp, n_fold))

    # 遍历独立实验
    for exp_id in range(n_exp):
        for fold_id in range(n_fold):
            fold_start_time = time.time()
            print(f"\nExp {exp_id+1}/{n_exp} | Fold {fold_id+1}/{n_fold}")

            if cancerType == 'pan-cancer':
                _, _, tr_mask, te_mask = k_sets[exp_id][fold_id]
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)
            else:
                tr_mask, te_mask = fixed_folds[fold_id]
                train_mask = torch.tensor(tr_mask).bool().to(device)
                test_mask = torch.tensor(te_mask).bool().to(device)

            # 初始化模型（单癌种默认 top_k=4, 泛癌种默认 top_k=5）
            model = combine_net_gate_without_ac(
                input_dim=input_dim,
                lambdinter=lambdinter,
                dropout=dropout,
                top_k=top_k
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            aurocs, auprcs, auroc, auprc = train_test(
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

            all_aurocs[:, exp_id, fold_id] = aurocs
            all_auprcs[:, exp_id, fold_id] = auprcs
            list_aurocs[exp_id, fold_id] = auroc  # 严格对应 AUROC
            list_auprcs[exp_id, fold_id] = auprc  # 严格对应 AUPRC

            fold_elapsed = time.time() - fold_start_time
            save_fold_result(run_dir, exp_id, fold_id, aurocs, auprcs, fold_elapsed)
            save_progress_snapshot(run_dir, all_aurocs, all_auprcs, list_aurocs, list_auprcs)

            if cancerType == 'pan-cancer':
                os.makedirs(RESULT_ROOT, exist_ok=True)
                np.savetxt(os.path.join(RESULT_ROOT, 'pan-cancer_auroc.txt'), list_aurocs, fmt='%.6f')
                np.savetxt(os.path.join(RESULT_ROOT, 'pan-cancer_auprc.txt'), list_auprcs, fmt='%.6f')
            else:
                single_result_dir = os.path.join(RESULT_ROOT, 'single')
                os.makedirs(single_result_dir, exist_ok=True)
                np.savetxt(
                    os.path.join(single_result_dir, f'{dataset}_{cancerType}_auroc.txt'),
                    list_aurocs,
                    fmt='%.6f'
                )
                np.savetxt(
                    os.path.join(single_result_dir, f'{dataset}_{cancerType}_auprc.txt'),
                    list_auprcs,
                    fmt='%.6f'
                )

    save_results_to_file(list_aurocs, list_auprcs, cancerType, dataset=dataset, lr=lr, dropout=dropout, lambdinter=lambdinter)

    save_best_epoch_results(
        list_aurocs, list_auprcs,
        all_aurocs, all_auprcs,
        cancerType=cancerType, dataset=dataset,
        lr=lr, dropout=dropout, lambdinter=lambdinter,
        epochs=epochs
    )

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
    return list_aurocs, list_auprcs


def main():
    parser = argparse.ArgumentParser(description="15 Single Cancers Training and Evaluation")
    parser.add_argument('--cancers', nargs='+', default=None,
                        help="指定评估癌种列表，默认运行配置文件中所有 15 个单癌种")
    parser.add_argument('--cancerType', type=str, default=None,
                        help="运行单个癌种 (例如 'blca')")
    parser.add_argument('--dataset', type=str, default='cpdb', choices=['cpdb', 'string'],
                        help="数据集类型 ('cpdb' 或 'string')")
    parser.add_argument('--epochs', type=int, default=160,
                        help="训练轮数 (默认 160)")
    parser.add_argument('--lr', type=float, default=0.0005,
                        help="学习率 (默认 0.0005)")
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout 率 (默认 0.3)")
    parser.add_argument('--lambdinter', type=float, default=0.001,
                        help="特征对齐损失权重 (默认 0.001)")
    parser.add_argument('--n_exp', type=int, default=10,
                        help="独立实验次数 (默认 10)")
    parser.add_argument('--n_fold', type=int, default=5,
                        help="折数 (默认 5)")
    parser.add_argument('--smoke_test', action='store_true',
                        help="快速冒烟测试 (1 次实验, 1 折, 2 个 epoch)")
    parser.add_argument('--split_file', type=str, default=None,
                        help="自定义单癌种 5 折划分文件路径 (.pkl)")

    args = parser.parse_args()

    if args.cancerType:
        cancers = [args.cancerType]
    elif args.cancers:
        cancers = args.cancers
    else:
        cancers = list(MOE_CONFIG['ENABLED_CANCERS'])

    n_exp = 1 if args.smoke_test else args.n_exp
    n_fold = 1 if args.smoke_test else args.n_fold
    epochs = 2 if args.smoke_test else args.epochs

    dataset = args.dataset

    for cancerType in cancers:
        if dataset == 'cpdb':
            data = torch.load(os.path.join(DATA_DIR, "CPDB", "CPDB_new_data.pt"))
            data = data.to(device)
            data.x = data.x[:, :48]
            if cancerType == 'pan-cancer':
                data.x = data.x[:, :48]
            else:
                cancerType_dict = MOE_CONFIG['CANCER_INDICES']
                data.x = data.x[:, cancerType_dict[cancerType]]

            datas = torch.load(os.path.join(DATA_DIR, "CPDB", "Str_feature.pkl")).to(device)
            data.x = torch.cat((data.x, datas), 1)
            data = data.to(device)

            with open(os.path.join(DATA_DIR, "CPDB", "k_sets.pkl"), 'rb') as handle:
                k_sets = pickle.load(handle)

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

        elif dataset == 'string':
            data = torch.load(os.path.join(DATA_DIR, "STRING", "STRING_data.pkl")).to(device)
            data.x = data.x[:, :48]

            datas = torch.load(os.path.join(DATA_DIR, "STRING", "Str_feature.pkl")).to(device)
            data.x = torch.cat((data.x, datas), 1)
            data = data.to(device)

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

        top_k = 5 if cancerType == 'pan-cancer' else 4

        print(f"\nTraining for cancer type: {cancerType}, input_dim: {input_dim}, top_k: {top_k}, dropout rate: {args.dropout}, learning rate: {args.lr}, lambda inter: {args.lambdinter}")

        trainPred_k_sets(
            input_dim=input_dim,
            k_sets=k_sets,
            data=data,
            L_emb=L_emb,
            edge_index=pb,
            L_emb_edge=L_emb_edge,
            lr=args.lr,
            epochs=epochs,
            lambdinter=args.lambdinter,
            dropout=args.dropout,
            cancerType=cancerType,
            dataset=dataset,
            n_exp=n_exp,
            n_fold=n_fold,
            top_k=top_k,
            split_file=args.split_file
        )


if __name__ == '__main__':
    main()
