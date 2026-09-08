import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import sys
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from sklearn import linear_model, metrics
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'baselines', 'mngcl'))

from alignment_check import assert_data_alignment
from gcn import GCN
from mngcl import MNGCL

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_mngcl_cv(args):
    # 0. 数据对齐严格断言
    assert_data_alignment(BASE_DIR)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*75}")
    print(f"Running MNGCL 10x5 Cross-Validation Reproduction (CPDB Pan-Cancer)")
    print(f"  Split Source   : data/CPDB/k_sets.pkl (Strict 10x5 folds)")
    print(f"  Protocol       : Transductive full graph, contrastive loss on all nodes,")
    print(f"                   classification loss on train nodes only, test once per fold (no early stopping/checkpoint)")
    print(f"  Runs / Folds   : {1 if args.smoke_test else args.n_runs} runs x {2 if args.smoke_test else 5} folds")
    print(f"  Epochs per fold: {5 if args.smoke_test else args.epochs}")
    print(f"  Hyperparameters: LR={args.lr}, tau=0.3, lambda=0.1")
    print(f"  Device         : {device}")
    print(f"{'='*75}\n")

    # 1. 种子初始化（采用 Bioprompt 原始种子 1234）
    fix_seed(args.seed)

    # 2. 读取 CPDB 多视图图数据与特征
    data_path = os.path.join(BASE_DIR, 'data', 'CPDB')
    data = torch.load(os.path.join(data_path, 'CPDB_data.pkl'), map_location=device)
    data.x = data.x[:, :48]

    dataz = torch.load(os.path.join(data_path, 'Str_feature.pkl'), map_location=device)
    data.x = torch.cat((data.x, dataz), 1).to(device)

    # Ground truth labels: y_all = y or y_te
    Y = torch.tensor(np.logical_or(data.y, data.y_te)).float().to(device)

    # 加载多视图网络及正样本矩阵
    ppiAdj = torch.load(os.path.join(data_path, 'ppi.pkl'), map_location='cpu')
    ppiAdj_self = torch.load(os.path.join(data_path, 'ppi_selfloop.pkl'), map_location='cpu')
    pathAdj = torch.load(os.path.join(data_path, 'pathway_SimMatrix.pkl'), map_location='cpu')
    goAdj = torch.load(os.path.join(data_path, 'GO_SimMatrix.pkl'), map_location='cpu')

    ppiAdj_index = ppiAdj.coalesce().indices().to(device)
    pathAdj_index = pathAdj.coalesce().indices().to(device)
    goAdj_index = goAdj.coalesce().indices().to(device)
    del ppiAdj

    if getattr(args, 'dense', False):
        print("Preparing original dense similarity matrices for contrastive loss (large VRAM mode)...")
        pos1 = ppiAdj_self.to_dense().to(device)
        del ppiAdj_self
        pos2 = pathAdj.to_dense().to(device)
        del pathAdj
        pos2.fill_diagonal_(1.0)
        pos3 = goAdj.to_dense().to(device)
        del goAdj
        pos3.fill_diagonal_(1.0)
        torch.cuda.empty_cache()
        posList = [pos1, pos2, pos3]
    else:
        print("Preparing sparse similarity matrices for contrastive loss (memory-optimized mode)...")
        n_nodes = data.x.shape[0]
        diag_indices = torch.arange(n_nodes, device=device).repeat(2, 1)
        diag_values = torch.ones(n_nodes, device=device)
        eye_sparse = torch.sparse_coo_tensor(diag_indices, diag_values, (n_nodes, n_nodes), device=device)

        pos1 = ppiAdj_self.to(device).coalesce()
        del ppiAdj_self

        pos2 = (pathAdj.to(device) + eye_sparse).coalesce()
        del pathAdj

        pos3 = (goAdj.to(device) + eye_sparse).coalesce()
        del goAdj

        torch.cuda.empty_cache()
        posList = [pos1, pos2, pos3]

    # 3. 读取严格 10x5 划分文件
    ksets_path = os.path.join(data_path, 'k_sets.pkl')
    assert os.path.exists(ksets_path), f"k_sets.pkl not found at {ksets_path}"
    with open(ksets_path, 'rb') as f:
        k_sets = pickle.load(f)

    n_runs = 1 if args.smoke_test else args.n_runs
    n_folds = 2 if args.smoke_test else 5
    epochs = min(args.epochs, 5) if args.smoke_test else args.epochs

    AUC = np.zeros((n_runs, n_folds))
    AUPR = np.zeros((n_runs, n_folds))

    LAMBDA = 0.1
    drop_edge_rate_1 = 0.2
    drop_edge_rate_2 = 0.1
    drop_edge_rate_3 = 0.1
    drop_feature_rate_1 = 0.5
    drop_feature_rate_2 = 0.3
    drop_feature_rate_3 = 0.4
    tau = 0.3

    total_start = time.time()

    for exp_id in range(n_runs):
        for fold_id in range(n_folds):
            fold_start = time.time()
            y_tr, y_val, tr_mask, te_mask = k_sets[exp_id][fold_id]
            train_mask = torch.tensor(tr_mask).bool().to(device)
            test_mask = torch.tensor(te_mask).bool().to(device)

            gcn = GCN(data.x.shape[1], 300, 100).to(device)
            model = MNGCL(
                gnn=gcn,
                posList=posList,
                tau=tau,
                gnn_outsize=100,
                projection_hidden_size=300,
                projection_size=100
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # 4. 训练阶段：对比损失使用全图，分类损失严格使用 tr_mask
            for epoch in range(1, epochs + 1):
                model.train()
                optimizer.zero_grad()

                x_1 = F.dropout(data.x, drop_feature_rate_1)
                x_2 = F.dropout(data.x, drop_feature_rate_2)
                x_3 = F.dropout(data.x, drop_feature_rate_3)

                aug_ppi = dropout_adj(ppiAdj_index, p=drop_edge_rate_1, force_undirected=True)[0]
                aug_path = dropout_adj(pathAdj_index, p=drop_edge_rate_2, force_undirected=True)[0]
                aug_go = dropout_adj(goAdj_index, p=drop_edge_rate_3, force_undirected=True)[0]

                pred1, pred2, pred3, _, conloss = model(aug_ppi, aug_path, aug_go, x_1, x_2, x_3)

                loss1 = F.binary_cross_entropy_with_logits(pred1[train_mask], Y[train_mask])
                loss2 = F.binary_cross_entropy_with_logits(pred2[train_mask], Y[train_mask])
                loss3 = F.binary_cross_entropy_with_logits(pred3[train_mask], Y[train_mask])

                crloss = LAMBDA * (loss1 + loss2 + loss3)
                loss = (1 - 3 * LAMBDA) * conloss + crloss
                loss.backward()
                optimizer.step()

                if epoch % 100 == 0 or epoch == epochs or epoch == 1:
                    print(f"    [Epoch {epoch:4d}/{epochs}] Loss: {loss.item():.4f} (contrastive: {conloss.item():.4f}, cls: {crloss.item():.4f})", flush=True)

            # 5. 测试阶段：训练结束后单次测试（按原实现训练 LogisticRegression 分类器）
            model.eval()
            with torch.no_grad():
                _, _, _, emb, _ = model(ppiAdj_index, pathAdj_index, goAdj_index, data.x, data.x, data.x)
                train_x = torch.sigmoid(emb[train_mask]).cpu().numpy()
                train_y = Y[train_mask].cpu().numpy().ravel()
                test_x = torch.sigmoid(emb[test_mask]).cpu().numpy()
                test_y = Y[test_mask].cpu().numpy().ravel()

                regr = linear_model.LogisticRegression(max_iter=10000)
                regr.fit(train_x, train_y)
                pred_prob = regr.predict_proba(test_x)[:, 1]

                fold_auroc = metrics.roc_auc_score(test_y, pred_prob)
                precision, recall, _ = metrics.precision_recall_curve(test_y, pred_prob)
                fold_auprc = metrics.auc(recall, precision)

                AUC[exp_id, fold_id] = fold_auroc
                AUPR[exp_id, fold_id] = fold_auprc

            fold_elapsed = time.time() - fold_start
            print(f"  >> [Exp {exp_id+1:02d}/{n_runs:02d} | Fold {fold_id+1}/{n_folds}] ({fold_elapsed:.1f}s) -> Test AUROC: {fold_auroc:.4f}, Test AUPRC: {fold_auprc:.4f}\n", flush=True)

            del model, optimizer, gcn
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    mean_auc, std_auc, var_auc = AUC.mean(), AUC.std(), AUC.var()
    mean_aupr, std_aupr, var_aupr = AUPR.mean(), AUPR.std(), AUPR.var()

    paper_auc = 0.9184
    paper_aupr = 0.8397

    print(f"\n{'='*75}")
    print(f"MNGCL 10x5 CV Final Reproduction Results (CPDB Pan-Cancer):")
    print(f"  AUROC : {mean_auc:.4f} ± {std_auc:.4f} (std), Variance: {var_auc:.6f}")
    print(f"  AUPRC : {mean_aupr:.4f} ± {std_aupr:.4f} (std), Variance: {var_aupr:.6f}")
    print(f"  Total Time: {total_time:.1f}s (Average {total_time / (n_runs * n_folds):.1f}s/fold)")
    print(f"Comparison with Paper (CPDB pan-cancer):")
    print(f"  Paper AUROC: {paper_auc:.4f} | Reproduced: {mean_auc:.4f} (Diff: {mean_auc - paper_auc:+.4f})")
    print(f"  Paper AUPRC: {paper_aupr:.4f} | Reproduced: {mean_aupr:.4f} (Diff: {mean_aupr - paper_aupr:+.4f})")
    print(f"{'='*75}\n")

    # 6. 保存结果文件（严格单独命名，避免覆盖 Clean/Hit）
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    prefix = "smoke_mngcl_cpdb_cv" if args.smoke_test else "mngcl_cpdb_cv"

    auroc_path = os.path.join(res_dir, f"{prefix}_auroc.txt")
    auprc_path = os.path.join(res_dir, f"{prefix}_auprc.txt")
    summary_path = os.path.join(res_dir, f"{prefix}_summary.txt")

    np.savetxt(auroc_path, AUC, fmt='%.4f', delimiter='\t')
    np.savetxt(auprc_path, AUPR, fmt='%.4f', delimiter='\t')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Method: MNGCL (Bioprompt Implementation)\n")
        f.write("Experiment: 10x5 Cross-Validation Reproduction (CPDB Pan-Cancer)\n")
        f.write("Split Source: data/CPDB/k_sets.pkl (Strict 10x5 folds, no random re-split)\n")
        f.write(f"Runs: {n_runs}, Folds: {n_folds}, Epochs: {epochs}, LR: {args.lr}\n")
        f.write("Protocol: Transductive full graph, contrastive loss on all nodes, classification loss on train nodes only, tested once after training\n")
        f.write("------------------------------------------------------------\n")
        f.write(f"AUROC Mean     : {mean_auc:.4f}\n")
        f.write(f"AUROC Std      : {std_auc:.4f}\n")
        f.write(f"AUROC Variance : {var_auc:.6f}\n")
        f.write(f"AUPRC Mean     : {mean_aupr:.4f}\n")
        f.write(f"AUPRC Std      : {std_aupr:.4f}\n")
        f.write(f"AUPRC Variance : {var_aupr:.6f}\n")
        f.write("------------------------------------------------------------\n")
        f.write("Paper Reference (CPDB):\n")
        f.write(f"  Paper AUROC: {paper_auc:.4f} | Diff: {mean_auc - paper_auc:+.4f}\n")
        f.write(f"  Paper AUPRC: {paper_aupr:.4f} | Diff: {mean_aupr - paper_aupr:+.4f}\n")
        f.write("------------------------------------------------------------\n")
        f.write("10x5 AUROC Matrix:\n")
        f.write(np.array2string(AUC, precision=4, suppress_small=True) + "\n\n")
        f.write("10x5 AUPRC Matrix:\n")
        f.write(np.array2string(AUPR, precision=4, suppress_small=True) + "\n")

    print(f"  Saved AUROC matrix to: {auroc_path}")
    print(f"  Saved AUPRC matrix to: {auprc_path}")
    print(f"  Saved Summary to: {summary_path}")

    return AUC, AUPR

def main():
    parser = argparse.ArgumentParser(description="MNGCL 10x5 CV Reproduction on CPDB k_sets.pkl")
    parser.add_argument('--smoke_test', action='store_true', help='Run quick 1-run 2-fold 5-epoch test')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dense', action='store_true', help='Use original dense similarity matrices for contrastive loss (requires >12GB GPU memory)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    run_mngcl_cv(args)

if __name__ == '__main__':
    main()
