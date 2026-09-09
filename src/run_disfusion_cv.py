import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import sys
import time
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))
sys.path.append(os.path.join(BASE_DIR, 'src', 'baselines', 'disfusion'))

from alignment_check import assert_data_alignment
from models import hypergrph_HGNN, graph_ChebNet, DISFusion
from utils import _generate_G_from_H_weight

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def fast_G_from_H_weight(H, W):
    """
    与 DISFusion 的 _generate_G_from_H_weight 数学上严格等价，
    优化避免显式创建巨大对角稀疏阵，极大提高构建速度。
    """
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE_W = W / DE
    H_scaled = H * invDE_W
    A = H_scaled @ H.T
    dv_inv_sqrt = np.power(DV, -0.5)[:, None]
    G = dv_inv_sqrt * A * dv_inv_sqrt.T
    return G

def get_incidence_matrix(data_dir, gene_list, use_pathway=True):
    """
    构建超图关联矩阵，过滤 cancer/tumor 相关 terms（遵循官方 utils.processingIncidenceMatrix）
    use_pathway=True: 载入 c2 (Curated Pathways) 与 c5 (Gene Ontology / HPO)
    use_pathway=False: 仅载入 c5 功能注释，不包含 c2 Pathway（论文基线设置）
    """
    ids = ['c2', 'c5'] if use_pathway else ['c5']
    incidenceMatrix = pd.DataFrame(index=gene_list)
    for id_name in ids:
        geneSetNameList = pd.read_csv(os.path.join(data_dir, f'{id_name}Name.txt'), sep='\t', header=None)
        geneSetNameList = list(geneSetNameList[0].values)
        idList = []
        for z, name in enumerate(geneSetNameList):
            if id_name == 'c2':
                q = name.split('_')
                if not ('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    idList.append(z)
            elif name[:2] == 'HP':
                q = name.split('_')
                if not ('CANCER' in q or 'TUMOR' in q or 'NEOPLASM' in q):
                    idList.append(z)
            else:
                idList.append(z)

        matrix = sp.load_npz(os.path.join(data_dir, f'{id_name}_GenesetsMatrix.npz'))
        matrix = pd.DataFrame(matrix.toarray(), index=gene_list)
        matrix = matrix.iloc[:, idList]
        incidenceMatrix = pd.concat([incidenceMatrix, matrix], axis=1)

    incidenceMatrix.columns = range(incidenceMatrix.shape[1])
    return incidenceMatrix

def run_disfusion_cv(args):
    # 0. 数据对齐严格断言
    assert_data_alignment(BASE_DIR)

    n_runs = 1 if args.smoke_test else args.n_runs
    n_folds = 2 if args.smoke_test else args.n_folds
    epochs = min(args.epochs, 5) if args.smoke_test else args.epochs

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*75}")
    print(f"Running DISFusion {n_runs}x{n_folds} Cross-Validation Reproduction (CPDB Pan-Cancer)")
    print(f"  Split Source   : data/CPDB/k_sets.pkl (Strict 10x5 folds)")
    print(f"  Pathway Used   : {args.use_pathway} ({'c2 Pathway + c5 Functional Annotations' if args.use_pathway else 'c5 Functional Annotations only (no c2 Pathway, Paper Table Setting)'})")
    print(f"  Protocol       : Transductive full graph, self-supervised Barlow Twins on all nodes,")
    print(f"                   disease-specific hypergraph weighted by train positive genes only,")
    print(f"                   classification loss on train nodes only, test once per fold (no early stopping)")
    print(f"  Runs / Folds   : {n_runs} runs x {n_folds} folds")
    print(f"  Epochs per fold: {epochs}")
    print(f"  Hyperparameters: lr=1e-5, weight_decay=5e-5, n_hid=256, lambdinter=1e-4, w_self=0.005")
    print(f"  Device         : {device}")
    print(f"{'='*75}\n")

    # 1. 种子初始化（采用官方原始种子 42）
    fix_seed(args.seed)

    # 2. 加载基础数据与超图
    dis_dir = os.path.join(BASE_DIR, 'data', 'DISFusion_data')
    gene_list_path = os.path.join(dis_dir, 'geneList.txt')
    gene_list = pd.read_csv(gene_list_path, header=None).iloc[:, 1].values
    gene_list = list(gene_list)

    print("Loading and filtering incidence matrix...")
    t0 = time.time()
    incidenceMatrix = get_incidence_matrix(dis_dir, gene_list, use_pathway=args.use_pathway)
    print(f"Incidence matrix ready in {time.time()-t0:.2f}s, shape: {incidenceMatrix.shape}")

    feat_path = os.path.join(dis_dir, 'biological features.csv')
    multi_feature = torch.Tensor(pd.read_csv(feat_path, sep=",").values).float().to(device)

    ppi_path = os.path.join(dis_dir, 'PPI_edge_index.txt')
    edge = np.array(np.loadtxt(ppi_path).transpose())
    PPI_graph = torch.from_numpy(edge).long().to(device)

    # 标签矩阵
    pos_path = os.path.join(dis_dir, '796true.txt')
    labelFrame = pd.DataFrame(data=[0] * len(gene_list), index=gene_list)
    positiveGene = pd.read_csv(pos_path, header=None)[0].values
    labelFrame.loc[positiveGene, :] = 1
    labels = torch.from_numpy(labelFrame.values.reshape(-1,)).long().to(device)

    # 3. 读取严格 10x5 划分文件
    ksets_path = os.path.join(BASE_DIR, 'data', 'CPDB', 'k_sets.pkl')
    assert os.path.exists(ksets_path), f"k_sets.pkl not found at {ksets_path}"
    with open(ksets_path, 'rb') as f:
        k_sets = pickle.load(f)

    AUC = np.zeros((n_runs, n_folds))
    AUPR = np.zeros((n_runs, n_folds))

    lr = args.lr
    weight_decay = 5e-5
    n_hid = 256
    lambdinter = 1e-4
    w_self = 0.005
    dropout = 0.5
    N = len(gene_list)

    total_start = time.time()

    for exp_id in range(n_runs):
        for fold_id in range(n_folds):
            fold_start = time.time()
            y_tr, y_val, tr_mask, te_mask = k_sets[exp_id][fold_id]
            trainIndex = np.where(tr_mask)[0]
            testIndex = np.where(te_mask)[0]

            # 4. 构建当前折特异性超图（仅使用当前折训练集阳性基因加权）
            trainFrame = labelFrame.iloc[trainIndex]
            trainPositiveGene = list(trainFrame.where(trainFrame == 1).dropna().index)
            positiveMatrixSum = incidenceMatrix.loc[trainPositiveGene].sum()

            selHyperedgeIndex = np.where(positiveMatrixSum >= 3)[0]
            selHyperedge = incidenceMatrix.iloc[:, selHyperedgeIndex]
            hyperedgeWeight = positiveMatrixSum[selHyperedgeIndex].values
            selHyperedgeWeightSum = incidenceMatrix.iloc[:, selHyperedgeIndex].values.sum(0)
            hyperedgeWeight = hyperedgeWeight / selHyperedgeWeightSum

            H = np.array(selHyperedge).astype('float')
            DV = np.sum(H * hyperedgeWeight, axis=1)
            for i in range(DV.shape[0]):
                if DV[i] == 0:
                    t_idx = random.randint(0, H.shape[1] - 1)
                    H[i][t_idx] = 0.0001

            if getattr(args, 'legacy_g', False):
                G = np.array(_generate_G_from_H_weight(H, hyperedgeWeight))
            else:
                G = fast_G_from_H_weight(H, hyperedgeWeight)
            adj_hyperGraph = torch.Tensor(G).float().to(device)
            fh = torch.eye(N).float().to(device)

            # 5. 初始化模型与优化器（严格遵循官方参数与结构）
            model_hypergrph = hypergrph_HGNN(in_ch=N, n_hid=n_hid, dropout=0.2).to(device)
            model_graph = graph_ChebNet(hdim=n_hid, dropout=0.5).to(device)
            optimizer_hypergrph = optim.Adam(model_hypergrph.parameters(), lr=0.005, weight_decay=5e-6)
            optimizer_graph = optim.Adam(model_graph.parameters(), lr=0.001, weight_decay=0)
            schedular_hypergrph = optim.lr_scheduler.MultiStepLR(optimizer_hypergrph, milestones=[100, 200, 300, 400], gamma=0.5)

            model_fusion = DISFusion(n_hid, 2, lambdinter, 0, 2, dropout=dropout).to(device)
            optimizer_fusion = optim.Adam(model_fusion.parameters(), lr=lr, weight_decay=weight_decay)

            # 6. 训练阶段：自监督损失作用于全图节点，分类损失严格作用于训练节点
            for epoch in range(epochs):
                model_hypergrph.train()
                model_graph.train()
                model_fusion.train()
                optimizer_hypergrph.zero_grad()
                optimizer_graph.zero_grad()
                optimizer_fusion.zero_grad()

                h_hyper = model_hypergrph(fh, adj_hyperGraph)
                h_ppi = model_graph(multi_feature, PPI_graph)
                loss_self, output_fusion = model_fusion(h_hyper, h_ppi)

                loss = F.nll_loss(output_fusion[trainIndex], labels[trainIndex])
                loss += w_self * loss_self
                loss.backward()

                optimizer_fusion.step()
                optimizer_hypergrph.step()
                optimizer_graph.step()
                schedular_hypergrph.step()

                if (epoch + 1) % 20 == 0 or epoch == 0 or (epoch + 1) == epochs:
                    print(f"    [Epoch {epoch+1:3d}/{epochs}] Loss: {loss.item():.4f} (self: {loss_self.item():.4f})", flush=True)

            # 7. 测试阶段：训练结束后对测试集进行单次评估
            model_hypergrph.eval()
            model_graph.eval()
            model_fusion.eval()
            with torch.no_grad():
                h_hyper = model_hypergrph(fh, adj_hyperGraph)
                h_ppi = model_graph(multi_feature, PPI_graph)
                _, output = model_fusion(h_hyper, h_ppi)

                outputTest = np.exp(output[testIndex].cpu().detach().numpy())[:, 1]
                labelsTest = labels[testIndex].cpu().numpy()

                fold_auroc = metrics.roc_auc_score(labelsTest, outputTest)
                precision, recall, _ = metrics.precision_recall_curve(labelsTest, outputTest)
                fold_auprc = metrics.auc(recall, precision)

                AUC[exp_id, fold_id] = fold_auroc
                AUPR[exp_id, fold_id] = fold_auprc

            fold_elapsed = time.time() - fold_start
            print(f"  >> [Exp {exp_id+1:02d}/{n_runs:02d} | Fold {fold_id+1}/{n_folds}] ({fold_elapsed:.1f}s) -> Test AUROC: {fold_auroc:.4f}, Test AUPRC: {fold_auprc:.4f}\n", flush=True)

            del model_hypergrph, model_graph, model_fusion, adj_hyperGraph, fh
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    mean_auc, std_auc, var_auc = AUC.mean(), AUC.std(), AUC.var()
    mean_aupr, std_aupr, var_aupr = AUPR.mean(), AUPR.std(), AUPR.var()

    paper_auc = 0.9238
    paper_aupr = 0.8436

    print(f"\n{'='*75}")
    print(f"DISFusion 10x5 CV Final Reproduction Results (CPDB Pan-Cancer):")
    print(f"  AUROC : {mean_auc:.4f} ± {std_auc:.4f} (std), Variance: {var_auc:.6f}")
    print(f"  AUPRC : {mean_aupr:.4f} ± {std_aupr:.4f} (std), Variance: {var_aupr:.6f}")
    print(f"  Total Time: {total_time:.1f}s (Average {total_time / (n_runs * n_folds):.1f}s/fold)")
    print(f"Comparison with Paper (CPDB pan-cancer):")
    print(f"  Paper AUROC: {paper_auc:.4f} | Reproduced: {mean_auc:.4f} (Diff: {mean_auc - paper_auc:+.4f})")
    print(f"  Paper AUPRC: {paper_aupr:.4f} | Reproduced: {mean_aupr:.4f} (Diff: {mean_aupr - paper_aupr:+.4f})")
    print(f"{'='*75}\n")

    # 8. 保存结果文件（严格单独命名，避免覆盖 Clean/Hit）
    res_dir = os.path.join(BASE_DIR, 'result')
    os.makedirs(res_dir, exist_ok=True)
    if args.smoke_test:
        prefix = f"smoke_disfusion_cpdb_cv{'' if args.use_pathway else '_nopathway'}"
    elif n_runs == 10 and n_folds == 5:
        prefix = f"disfusion_cpdb_cv{'' if args.use_pathway else '_nopathway'}"
    else:
        prefix = f"quick_disfusion_cpdb_cv{'' if args.use_pathway else '_nopathway'}_{n_runs}x{n_folds}"

    auroc_path = os.path.join(res_dir, f"{prefix}_auroc.txt")
    auprc_path = os.path.join(res_dir, f"{prefix}_auprc.txt")
    summary_path = os.path.join(res_dir, f"{prefix}_summary.txt")

    np.savetxt(auroc_path, AUC, fmt='%.4f', delimiter='\t')
    np.savetxt(auprc_path, AUPR, fmt='%.4f', delimiter='\t')

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Method: DISFusion (Official Implementation)\n")
        f.write("Experiment: 10x5 Cross-Validation Reproduction (CPDB Pan-Cancer)\n")
        f.write("Split Source: data/CPDB/k_sets.pkl (Strict 10x5 folds, no random re-split)\n")
        f.write(f"Use Pathway: {args.use_pathway} ({'c2 + c5' if args.use_pathway else 'c5 only (no c2 pathway, Paper Table Setting)'})\n")
        f.write(f"Runs: {n_runs}, Folds: {n_folds}, Epochs: {epochs}, LR: {lr}\n")
        f.write("Protocol: Transductive full graph, self-supervised Barlow Twins on all nodes, disease-specific hypergraph weighted by train positive genes only, classification loss on train nodes only, tested once after training\n")
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
        f.write(f"{n_runs}x{n_folds} AUROC Matrix:\n")
        f.write(np.array2string(AUC, precision=4, suppress_small=True) + "\n\n")
        f.write(f"{n_runs}x{n_folds} AUPRC Matrix:\n")
        f.write(np.array2string(AUPR, precision=4, suppress_small=True) + "\n")

    print(f"  Saved AUROC matrix to: {auroc_path}")
    print(f"  Saved AUPRC matrix to: {auprc_path}")
    print(f"  Saved Summary to: {summary_path}")

    return AUC, AUPR

def main():
    parser = argparse.ArgumentParser(description="DISFusion 10x5 CV Reproduction on CPDB k_sets.pkl")
    parser.add_argument('--smoke_test', action='store_true', help='Run quick 1-run 2-fold 5-epoch test')
    parser.add_argument('--use_pathway', type=str2bool, default=True, help='Whether to include c2 pathway hyperedges (default: True)')
    parser.add_argument('--no_pathway', dest='use_pathway', action='store_false', help='Disable c2 pathway hyperedges (use only c5 functional annotations, paper table setting)')
    parser.add_argument('--n_runs', type=int, default=10, help='Number of repeat runs (default: 10)')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds per run (1-5, default: 5)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--legacy_g', action='store_true', help='Use original slow _generate_G_from_H_weight (dense diagonal matrix inversion)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    run_disfusion_cv(args)

if __name__ == '__main__':
    main()
