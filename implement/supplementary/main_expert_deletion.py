"""Run held-out-gene expert-deletion faithfulness experiments for BIMODriver."""

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'BIMODriver') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'BIMODriver'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn import metrics
from torch_geometric.utils import (
    add_self_loops, dropout_adj, remove_self_loops, subgraph,
)

from model_expert_deletion import combine_net_gate_without_ac


EXPERTS = ("Omics", "Self", "Neighbor", "Together", "Interact_OS", "Interact_ST")
CSV_FIELDS = (
    "protocol", "model", "seed", "fold", "split", "gene_index", "gene_id",
    "class", "expert", "expert_logit", "gate_logit", "gate_probability",
    "topk_selected", "top_k",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Top-K expert-deletion faithfulness tests on held-out genes"
    )
    parser.add_argument("--protocol", choices=("transductive", "inductive"),
                        default="transductive")
    parser.add_argument("--experiments", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lambda-inter", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_strict_inductive_training_view(
        features, language_embeddings, ppi_edges, semantic_edges, test_mask):
    """Remove test nodes and return reindexed training inputs and graphs.

    Unlike edge masking with zero-valued placeholders, this constructs a real
    induced subgraph. Test genes therefore do not occur in any tensor passed to
    the model during optimization.
    """
    test_mask = test_mask.to(device=features.device, dtype=torch.bool)
    keep_nodes = (~test_mask).nonzero(as_tuple=True)[0]
    if keep_nodes.numel() == 0:
        raise ValueError("strict-inductive training subgraph cannot be empty")
    # Zero held-out rows before subsetting as an explicit leakage safeguard.
    # The subsequent index_select then removes those rows from the model input
    # altogether, satisfying both zero-feature and absent-node requirements.
    held_out_rows = test_mask.reshape(-1, 1)
    zero_masked_features = features.masked_fill(held_out_rows, 0.0)
    zero_masked_language = {
        name: value.masked_fill(held_out_rows, 0.0)
        for name, value in language_embeddings.items()
    }
    ppi_train, _ = subgraph(
        keep_nodes, ppi_edges, relabel_nodes=True, num_nodes=features.size(0)
    )
    semantic_train, _ = subgraph(
        keep_nodes, semantic_edges, relabel_nodes=True, num_nodes=features.size(0)
    )
    training_features = zero_masked_features.index_select(0, keep_nodes)
    training_language = {
        name: value.index_select(0, keep_nodes)
        for name, value in zero_masked_language.items()
    }
    node_count = keep_nodes.numel()
    for name, edges in (("CPDB", ppi_train), ("semantic KNN", semantic_train)):
        if edges.numel() and int(edges.max()) >= node_count:
            raise RuntimeError(f"{name} induced graph was not correctly reindexed")
    return (training_features, training_language, ppi_train, semantic_train,
            keep_nodes)


def get_class_weight(labels):
    positive = labels.sum(dim=0)
    negative = labels.shape[0] - positive
    return negative / (positive + 1e-6)


def compute_loss(outputs, indices, labels, lambda_inter):
    loss_inter, *expert_and_fused = outputs
    class_weight = get_class_weight(labels[indices])
    classification = sum(
        F.binary_cross_entropy_with_logits(
            logits[indices], labels[indices], pos_weight=class_weight
        )
        for logits in expert_and_fused
    )
    return classification + lambda_inter * loss_inter


def evaluate(outputs, test_mask, labels):
    truth = labels[test_mask].detach().cpu().numpy().ravel()
    prediction = torch.sigmoid(outputs[-1][test_mask]).detach().cpu().numpy().ravel()
    precision, recall, _ = metrics.precision_recall_curve(truth, prediction)
    return {
        "auroc": float(metrics.roc_auc_score(truth, prediction)),
        "auprc": float(metrics.auc(recall, precision)),
    }


def load_gene_ids(path, number_of_nodes):
    if path.exists():
        values = [line.strip().split()[0]
                  for line in path.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        if len(values) == number_of_nodes:
            return values
        print(f"Warning: {path} contains {len(values)} names; using node indices.")
    return [str(index) for index in range(number_of_nodes)]


def export_split(writer, routing, mask, labels, gene_ids, protocol, seed, fold, split):
    indices = mask.nonzero(as_tuple=True)[0].detach().cpu().numpy()
    expert_logits = routing["expert_logits"].detach().cpu().numpy()
    gate_logits = routing["gate_logits"].detach().cpu().numpy()
    probabilities = routing["gate_probability"].detach().cpu().numpy()
    selected = routing["topk_selected"].detach().cpu().numpy()
    targets = labels.detach().cpu().numpy().reshape(-1)

    for gene_index in indices:
        class_name = "Driver" if targets[gene_index] >= 0.5 else "Non-driver"
        for expert_index, expert_name in enumerate(EXPERTS):
            writer.writerow({
                "protocol": protocol, "model": "sparse_top4", "seed": seed,
                "fold": fold, "split": split, "gene_index": int(gene_index),
                "gene_id": gene_ids[gene_index], "class": class_name,
                "expert": expert_name,
                "expert_logit": float(expert_logits[gene_index, expert_index]),
                "gate_logit": float(gate_logits[gene_index, expert_index]),
                "gate_probability": float(probabilities[gene_index, expert_index]),
                "topk_selected": int(selected[gene_index, expert_index]),
                "top_k": 4,
            })


def score_predictions(truth, probability):
    precision, recall, _ = metrics.precision_recall_curve(truth, probability)
    return (float(metrics.roc_auc_score(truth, probability)),
            float(metrics.auc(recall, precision)))


def deletion_outputs(model, routing, test_mask, labels, gene_ids, seed, fold,
                     threshold, random_seed):
    """Generate gene-level and fold-level results for every deletion rule."""
    node_indices = test_mask.nonzero(as_tuple=True)[0]
    expert_logits = routing["expert_logits"][node_indices]
    weights = routing["gate_probability"][node_indices]
    selected = routing["topk_selected"][node_indices]
    original = torch.sigmoid((expert_logits * weights).sum(1)).cpu().numpy()
    truth = labels[node_indices].cpu().numpy().reshape(-1)
    weights_np, selected_np = weights.cpu().numpy(), selected.cpu().numpy()
    rng = np.random.default_rng(random_seed)
    chosen = {
        "remove_highest": weights_np.argmax(1),
        "remove_lowest": np.where(selected_np, weights_np, np.inf).argmin(1),
        "remove_random": np.array([
            rng.choice(np.flatnonzero(row)) for row in selected_np
        ]),
    }
    probabilities = {}
    removed_names = {}
    for intervention, indices in chosen.items():
        changed = model.predict_after_deletion(expert_logits, weights, indices)
        probabilities[intervention] = torch.sigmoid(changed).cpu().numpy().reshape(-1)
        removed_names[intervention] = [EXPERTS[int(index)] for index in indices]
    for expert_index, expert_name in enumerate(EXPERTS):
        intervention = f"remove_fixed_{expert_name}"
        indices = np.where(selected_np[:, expert_index], expert_index, -1)
        changed = model.predict_after_deletion(expert_logits, weights, indices)
        probabilities[intervention] = torch.sigmoid(changed).cpu().numpy().reshape(-1)

    base_auc, base_auprc = score_predictions(truth, original)
    fold_rows, class_rows = [], []
    for intervention, changed in probabilities.items():
        changed_auc, changed_auprc = score_predictions(truth, changed)
        fold_rows.append({
            "seed": seed, "fold": fold, "intervention": intervention,
            "mean_abs_probability_change": np.abs(changed - original).mean(),
            "median_abs_probability_change": np.median(np.abs(changed - original)),
            "flip_rate": np.mean((original >= threshold) != (changed >= threshold)),
            "original_auroc": base_auc, "intervention_auroc": changed_auc,
            "auroc_drop": base_auc - changed_auc,
            "original_auprc": base_auprc, "intervention_auprc": changed_auprc,
            "auprc_drop": base_auprc - changed_auprc,
        })
        for class_value, class_name in ((1, "Driver"), (0, "Non-driver")):
            class_mask = truth == class_value
            class_rows.append({
                "seed": seed, "fold": fold, "class": class_name,
                "intervention": intervention, "n_genes": int(class_mask.sum()),
                "mean_abs_probability_change": np.abs(
                    changed[class_mask] - original[class_mask]
                ).mean(),
                "flip_rate": np.mean(
                    (original[class_mask] >= threshold) !=
                    (changed[class_mask] >= threshold)
                ),
            })

    gene_rows = []
    for local_index, node_index in enumerate(node_indices.cpu().numpy()):
        row = {
            "seed": seed, "fold": fold, "gene_index": int(node_index),
            "gene_id": gene_ids[node_index], "label": int(truth[local_index]),
            "original_probability": float(original[local_index]),
            "original_prediction": int(original[local_index] >= threshold),
            "topk_experts": ";".join(
                EXPERTS[j] for j in np.flatnonzero(selected_np[local_index])
            ),
        }
        for j, expert in enumerate(EXPERTS):
            row[f"weight_{expert}"] = float(weights_np[local_index, j])
        for intervention in ("remove_highest", "remove_lowest", "remove_random"):
            row[f"{intervention}_expert"] = removed_names[intervention][local_index]
            row[f"{intervention}_probability"] = float(
                probabilities[intervention][local_index]
            )
        for expert in EXPERTS:
            row[f"remove_fixed_{expert}_probability"] = float(
                probabilities[f"remove_fixed_{expert}"][local_index]
            )
        gene_rows.append(row)
    return gene_rows, fold_rows, class_rows


def rank_biserial(differences):
    differences = np.asarray(differences, float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() -
                  ranks[differences < 0].sum()) / ranks.sum())


def statistical_comparison(frame, samples, random_seed):
    rows, rng = [], np.random.default_rng(random_seed)
    dynamic = frame[frame.intervention.isin(
        ["remove_highest", "remove_lowest", "remove_random"]
    )]
    for measure in ("mean_abs_probability_change", "flip_rate",
                    "auroc_drop", "auprc_drop"):
        pivot = dynamic.pivot(index=["seed", "fold"], columns="intervention",
                              values=measure).dropna()
        differences = (pivot.remove_highest - pivot.remove_lowest).to_numpy()
        if len(differences) == 0 or np.allclose(differences, 0):
            statistic, p_value = 0.0, 1.0
        else:
            test = stats.wilcoxon(differences, alternative="two-sided")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        seed_differences = (pivot.remove_highest - pivot.remove_lowest).groupby(
            level="seed"
        ).mean().to_numpy()
        if len(seed_differences) > 1:
            bootstrap = rng.choice(
                seed_differences, (samples, len(seed_differences)), replace=True
            ).mean(1)
            ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
        else:
            ci_low = ci_high = np.nan
        rows.append({
            "metric": measure, "comparison": "highest_minus_lowest",
            "fold_pairs": len(differences), "seed_count": len(seed_differences),
            "mean_paired_difference": np.mean(differences),
            "median_paired_difference": np.median(differences),
            "wilcoxon_statistic": statistic, "wilcoxon_p_two_sided": p_value,
            "matched_rank_biserial": rank_biserial(differences),
            "across_seed_mean_difference": np.mean(seed_differences),
            "across_seed_bootstrap_ci95_low": ci_low,
            "across_seed_bootstrap_ci95_high": ci_high,
        })
    return pd.DataFrame(rows)


def load_cpdb(repo_dir, device):
    data_dir = repo_dir / "data" / "CPDB"
    data = torch.load(data_dir / "CPDB_new_data.pt", map_location=device).to(device)
    data.x = data.x[:, :48]
    structural = torch.load(data_dir / "Str_feature.pkl", map_location=device).to(device)
    data.x = torch.cat((data.x, structural), dim=1).detach()
    with (data_dir / "k_sets.pkl").open("rb") as handle:
        split_sets = pickle.load(handle)

    statement = torch.load(
        data_dir / "PAN-CANCER_statement_features.pt", map_location=device
    ).to(device)
    language_embeddings = {
        "self_emb": statement[:, 0:768].detach(),
        "neighbor_emb": statement[:, 768:1536].detach(),
        "together_emb": statement[:, 1536:2304].detach(),
    }
    semantic_edges = torch.load(
        repo_dir / "data" / "cpdb_network_LLM" / "merged_k5_edge_index.pt",
        map_location=device,
    ).to(device)
    ppi_edges, _ = remove_self_loops(data.edge_index)
    ppi_edges, _ = add_self_loops(ppi_edges)
    labels = torch.logical_or(
        torch.as_tensor(data.y, device=device).bool(),
        torch.as_tensor(data.y_te, device=device).bool(),
    ).to(dtype=torch.float32)
    if labels.ndim == 1:
        labels = labels.unsqueeze(1)
    gene_ids = load_gene_ids(data_dir / "node_names.txt", data.num_nodes)
    return data, split_sets, language_embeddings, semantic_edges, ppi_edges, labels, gene_ids


def main():
    args = parse_args()
    if not 1 <= args.experiments <= 10 or not 1 <= args.folds <= 5:
        raise ValueError("--experiments must be 1..10 and --folds must be 1..5")

    output = args.output or (
        REPO_ROOT / "outputs" / "expert_deletion" / "CPDB" / "cpdb" /
        args.protocol / "oof_gene_routing.csv"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = output.with_name("metrics.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_cpdb(REPO_ROOT, device)
    data, split_sets, language_embeddings, semantic_edges, ppi_edges, labels, gene_ids = loaded

    metric_rows, gene_rows, deletion_rows, class_rows = [], [], [], []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for experiment in range(args.experiments):
            run_seed = args.seed + experiment
            for fold_index in range(args.folds):
                set_seed(run_seed * 100 + fold_index)
                _, _, train_raw, test_raw = split_sets[experiment][fold_index]
                train_mask = torch.as_tensor(train_raw, dtype=torch.bool, device=device)
                test_mask = torch.as_tensor(test_raw, dtype=torch.bool, device=device)
                if torch.any(train_mask & test_mask):
                    raise RuntimeError("train and test masks overlap")

                if args.protocol == "inductive":
                    (training_features, training_language, ppi_train,
                     semantic_train, keep_nodes) = make_strict_inductive_training_view(
                        data.x, language_embeddings, ppi_edges, semantic_edges,
                        test_mask,
                    )
                    # keep_nodes is sorted; both labels and the supervised mask
                    # are reindexed into exactly the same induced-node order.
                    training_labels = labels.index_select(0, keep_nodes)
                    train_indices = train_mask.index_select(0, keep_nodes).nonzero(
                        as_tuple=True
                    )[0]
                    if training_features.size(0) != int((~test_mask).sum()):
                        raise RuntimeError("a held-out node entered the training view")
                else:
                    training_features = data.x
                    training_language = language_embeddings
                    training_labels = labels
                    ppi_train, semantic_train = ppi_edges, semantic_edges
                    train_indices = train_mask.nonzero(as_tuple=True)[0]

                model = combine_net_gate_without_ac(
                    input_dim=data.x.shape[1], lambdinter=args.lambda_inter,
                    dropout=args.dropout,
                ).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

                for epoch in range(args.epochs):
                    model.train()
                    optimizer.zero_grad()
                    dropped_ppi = dropout_adj(ppi_train, p=0.3)[0]
                    outputs = model(
                        training_features, dropped_ppi, training_language,
                        semantic_train,
                    )
                    loss = compute_loss(
                        outputs, train_indices, training_labels, args.lambda_inter
                    )
                    loss.backward()
                    optimizer.step()
                    if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
                        print(
                            f"protocol={args.protocol} seed={experiment + 1}/{args.experiments} "
                            f"fold={fold_index + 1}/{args.folds} epoch={epoch + 1}/{args.epochs} "
                            f"loss={loss.item():.6f}"
                        )

                model.eval()
                with torch.no_grad():
                    # Restore the complete graphs and the original precomputed
                    # features only after optimization has finished.
                    test_outputs, test_routing = model(
                        data.x, ppi_edges, language_embeddings, semantic_edges,
                        return_routing=True,
                    )

                export_split(writer, test_routing, test_mask, labels, gene_ids,
                             args.protocol, run_seed, fold_index + 1, "test")
                row = evaluate(test_outputs, test_mask, labels)
                row.update({"seed": run_seed, "fold": fold_index + 1})
                metric_rows.append(row)
                genes, deletions, classes = deletion_outputs(
                    model, test_routing, test_mask, labels, gene_ids,
                    run_seed, fold_index + 1, args.threshold,
                    random_seed=run_seed * 10000 + fold_index,
                )
                gene_rows.extend(genes)
                deletion_rows.extend(deletions)
                class_rows.extend(classes)
                # Incremental writes preserve completed folds if a long run stops.
                pd.DataFrame(gene_rows).to_csv(
                    output.with_name("held_out_gene_predictions.csv"), index=False
                )
                pd.DataFrame(deletion_rows).to_csv(
                    output.with_name("deletion_fold_metrics_overall.csv"), index=False
                )
                pd.DataFrame(class_rows).to_csv(
                    output.with_name("deletion_fold_metrics_by_class.csv"), index=False
                )
                handle.flush()

    summary = {
        "protocol": args.protocol, "top_k": 4,
        "evaluation_epoch": args.epochs,
        "epoch_selection": "final_epoch",
        "probability_definition": (
            "softmax over selected Top-4 gate logits; other two experts are zero"
        ),
        "experiments": args.experiments, "folds": args.folds,
        "epochs": args.epochs, "metrics": metric_rows,
        "mean_auroc": float(np.mean([row["auroc"] for row in metric_rows])),
        "mean_auprc": float(np.mean([row["auprc"] for row in metric_rows])),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    deletion_frame = pd.DataFrame(deletion_rows)
    comparison = statistical_comparison(
        deletion_frame, args.bootstrap_samples, args.seed
    )
    comparison.to_csv(
        output.with_name("paired_wilcoxon_highest_vs_lowest.csv"), index=False
    )
    deletion_frame[
        deletion_frame.intervention.str.startswith("remove_fixed_")
    ].to_csv(output.with_name("fixed_expert_deletion_fold_metrics.csv"), index=False)
    metadata = {
        "protocol": args.protocol,
        "strict_inductive_training": {
            "test_feature_rows_zeroed_before_subsetting": args.protocol == "inductive",
            "test_nodes_removed_from_model_inputs": args.protocol == "inductive",
            "cpdb_and_semantic_graphs_induced_and_reindexed": args.protocol == "inductive",
            "test_nodes_excluded_from_contrastive_loss": args.protocol == "inductive",
            "test_labels_excluded_from_supervised_loss": True,
            "full_features_and_graphs_restored_only_for_inference": args.protocol == "inductive",
            "parameter_updates_during_inference": False,
        },
        "experts": list(EXPERTS),
        "top_k": 4,
        "renormalize_remaining_weights": True,
        "random_baseline": "one uniformly sampled selected expert per held-out gene",
        "prediction_threshold": args.threshold,
        "statistical_unit": "fold",
        "wilcoxon": "paired two-sided, remove_highest versus remove_lowest",
        "effect_size": "matched-pairs rank-biserial correlation",
        "ci95": "percentile bootstrap of seed-level mean paired differences",
        "bootstrap_samples": args.bootstrap_samples,
        "base_training_seed": args.seed,
    }
    output.with_name("experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Routing CSV: {output}")
    print(f"Metrics JSON: {summary_path}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
