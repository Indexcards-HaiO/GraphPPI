#!/usr/bin/env python3
"""
Node ranking for GraphPPI.

This module reuses the link-prediction model to score candidate genes by their
predicted interaction strength with a set of seed genes. It is written to work
with the current small breast-cancer graph and with larger graphs that follow
the same preprocessed PyG Data format.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from graphppi.models.predictor import LinkPredictor
    from graphppi.trainer import LinkPredictionTrainer
    from graphppi.utils import (
        compute_features_from_edges,
        sample_negative_edges,
    )
except ModuleNotFoundError:
    # Allow `python src/rank_genes.py` before `pip install -e .`.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.predictor import LinkPredictor
    from trainer import LinkPredictionTrainer
    from utils import (
        compute_features_from_edges,
        sample_negative_edges,
    )


DEFAULT_SEEDS = ["TP53", "BRCA1", "ERBB2", "PIK3CA", "ESR1"]


def load_name_list(values=None, filepath=None):
    """Load node names from command-line values and/or a one-column text file."""
    names = []
    if values:
        for value in values:
            names.extend([item.strip() for item in value.split(",") if item.strip()])

    if filepath:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    names.append(value.split()[0])

    return list(dict.fromkeys(names))


def get_undirected_edges(data):
    """Return one directed representation of each stored undirected edge."""
    num_undirected = data.edge_index.size(1) // 2
    edge_index = data.edge_index[:, :num_undirected]
    edge_weight = data.edge_weight[:num_undirected]
    edge_attr = data.edge_attr[:num_undirected] if hasattr(data, "edge_attr") else None
    return edge_index, edge_weight, edge_attr


def build_observed_edge_set(edge_index):
    """Build an undirected lookup set for known edges."""
    observed = set()
    for i in range(edge_index.size(1)):
        u = int(edge_index[0, i].item())
        v = int(edge_index[1, i].item())
        observed.add((u, v))
        observed.add((v, u))
    return observed


def make_bidirectional(edge_index, edge_weight):
    """Create a bidirectional edge list for undirected feature calculation."""
    rev_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    return torch.cat([edge_index, rev_edge_index], dim=1), torch.cat([edge_weight, edge_weight], dim=0)


def split_train_val_edges(edge_index, edge_weight, edge_attr=None, val_fraction=0.15, seed=42):
    """Split observed undirected edges into train and validation positives."""
    num_edges = edge_index.size(1)
    if num_edges < 2:
        raise ValueError("Need at least two observed edges to train and validate ranking model.")

    rng = np.random.default_rng(seed)
    indices = np.arange(num_edges)
    rng.shuffle(indices)

    val_size = max(1, int(num_edges * val_fraction))
    val_size = min(val_size, num_edges - 1)
    val_idx = torch.tensor(indices[:val_size], dtype=torch.long)
    train_idx = torch.tensor(indices[val_size:], dtype=torch.long)

    result = {
        "train_pos": edge_index[:, train_idx],
        "val_pos": edge_index[:, val_idx],
        "train_weight": edge_weight[train_idx],
        "val_weight": edge_weight[val_idx],
    }
    if edge_attr is not None:
        result["train_attr"] = edge_attr[train_idx]
        result["val_attr"] = edge_attr[val_idx]
    else:
        result["train_attr"] = None
        result["val_attr"] = None
    return result


def make_edge_attr(pos_attr, neg_count, edge_attr_dim=8):
    """Concatenate positive edge attributes with zero attributes for sampled non-edges."""
    if pos_attr is None:
        return None
    zeros = torch.zeros(neg_count, edge_attr_dim, dtype=pos_attr.dtype)
    return torch.cat([pos_attr, zeros], dim=0)


def train_ranking_model(data, config):
    """
    Train a link-prediction model for downstream node ranking.

    Validation is held out for early stopping. After training, ranking uses the
    full observed graph for message passing so the final node scores can use all
    available known interactions.
    """
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    node_names = list(data.node_names)
    full_edge_index, full_edge_weight, full_edge_attr = get_undirected_edges(data)
    split = split_train_val_edges(
        full_edge_index,
        full_edge_weight,
        full_edge_attr,
        val_fraction=config["val_fraction"],
        seed=config["seed"],
    )

    exclude_set = build_observed_edge_set(full_edge_index)
    train_neg = sample_negative_edges(
        split["train_pos"],
        data.num_nodes,
        split["train_pos"].size(1) * config["neg_ratio"],
        exclude_set.copy(),
        random_state=config["seed"],
    )
    val_neg = sample_negative_edges(
        split["val_pos"],
        data.num_nodes,
        split["val_pos"].size(1) * config["neg_ratio"],
        exclude_set.copy(),
        random_state=config["seed"] + 1,
    )

    train_edges = torch.cat([split["train_pos"], train_neg], dim=1)
    train_labels = torch.cat([torch.ones(split["train_pos"].size(1)), torch.zeros(train_neg.size(1))])
    val_edges = torch.cat([split["val_pos"], val_neg], dim=1)
    val_labels = torch.cat([torch.ones(split["val_pos"].size(1)), torch.zeros(val_neg.size(1))])

    train_edge_attr = make_edge_attr(split["train_attr"], train_neg.size(1), config["edge_attr_dim"])
    val_edge_attr = make_edge_attr(split["val_attr"], val_neg.size(1), config["edge_attr_dim"])

    train_mp_edge_index, train_mp_edge_weight = make_bidirectional(
        split["train_pos"], split["train_weight"]
    )
    train_feature_edges, train_feature_weights = make_bidirectional(
        split["train_pos"], split["train_weight"]
    )
    x = compute_features_from_edges(
        train_feature_edges,
        train_feature_weights,
        data.num_nodes,
        node_names,
        seed_genes=config["seed_genes"],
    )
    in_dim = x.size(1)
    if config["use_identity"]:
        x = torch.eye(data.num_nodes)
        in_dim = data.num_nodes

    model = LinkPredictor(
        in_dim=in_dim,
        encoder_type=config["encoder"],
        decoder_type=config["decoder"],
        hidden_dim=config["hidden_dim"],
        out_dim=config["out_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )
    trainer = LinkPredictionTrainer(
        model,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        device=config["device"],
    )
    trainer.train(
        x=x,
        mp_edge_index=train_mp_edge_index,
        train_edges=train_edges,
        train_labels=train_labels,
        val_edges=val_edges,
        val_labels=val_labels,
        epochs=config["epochs"],
        patience=config["patience"],
        mp_edge_weight=train_mp_edge_weight,
        train_edge_attr=train_edge_attr,
        val_edge_attr=val_edge_attr,
        verbose=config["verbose"],
        log_interval=config["log_interval"],
    )

    full_feature_edges, full_feature_weights = make_bidirectional(
        full_edge_index, full_edge_weight
    )
    full_x = compute_features_from_edges(
        full_feature_edges,
        full_feature_weights,
        data.num_nodes,
        node_names,
        seed_genes=config["seed_genes"],
    )
    if config["use_identity"]:
        full_x = torch.eye(data.num_nodes)

    full_mp_edge_index, full_mp_edge_weight = make_bidirectional(
        full_edge_index, full_edge_weight
    )

    return {
        "model": trainer.get_model(),
        "x": full_x,
        "mp_edge_index": full_mp_edge_index,
        "mp_edge_weight": full_mp_edge_weight,
        "observed_edges": exclude_set,
        "node_names": node_names,
        "best_val_auc": trainer.best_val_auc,
    }


def aggregate_scores(scores, mode):
    """Aggregate one candidate's seed-link scores."""
    if not scores:
        return {
            "score": np.nan,
            "mean_score": np.nan,
            "max_score": np.nan,
            "min_score": np.nan,
            "num_scored_seeds": 0,
        }
    arr = np.array(scores, dtype=float)
    values = {
        "mean_score": float(arr.mean()),
        "max_score": float(arr.max()),
        "min_score": float(arr.min()),
        "num_scored_seeds": int(arr.size),
    }
    values["score"] = values[f"{mode}_score"] if mode in {"mean", "max", "min"} else values["mean_score"]
    return values


def rank_candidates(model_bundle, seed_genes, candidate_genes=None, top_k=20,
                    batch_size=8192, aggregate="mean", exclude_known=True,
                    device="cpu"):
    """Score candidate nodes against seed nodes in batches."""
    node_names = model_bundle["node_names"]
    name_to_idx = {name: i for i, name in enumerate(node_names)}

    seed_pairs = [(name, name_to_idx[name]) for name in seed_genes if name in name_to_idx]
    seed_indices = [idx for _, idx in seed_pairs]
    missing_seeds = [name for name in seed_genes if name not in name_to_idx]
    if not seed_indices:
        raise ValueError("None of the requested seed genes exist in the graph.")

    if candidate_genes:
        candidates = [name for name in candidate_genes if name in name_to_idx]
    else:
        seed_set = set(seed_genes)
        candidates = [name for name in node_names if name not in seed_set]

    model = model_bundle["model"].to(device)
    model.eval()
    x = model_bundle["x"].to(device)
    mp_edge_index = model_bundle["mp_edge_index"].to(device)
    mp_edge_weight = model_bundle["mp_edge_weight"].to(device)
    observed = model_bundle["observed_edges"]

    rows = []
    with torch.no_grad():
        z = model.encode(x, mp_edge_index, mp_edge_weight)

        for candidate_name in candidates:
            candidate_idx = name_to_idx[candidate_name]
            pair_list = []
            scored_seed_names = []
            for seed_name, seed_idx in seed_pairs:
                if candidate_idx == seed_idx:
                    continue
                if exclude_known and (candidate_idx, seed_idx) in observed:
                    continue
                pair_list.append((candidate_idx, seed_idx))
                scored_seed_names.append(seed_name)

            scores = []
            for start in range(0, len(pair_list), batch_size):
                batch = pair_list[start:start + batch_size]
                if not batch:
                    continue
                edge_index = torch.tensor(batch, dtype=torch.long, device=device).t().contiguous()
                pred = model.decode(z, edge_index, None)
                scores.extend(pred.detach().cpu().numpy().tolist())

            agg = aggregate_scores(scores, aggregate)
            rows.append({
                "gene": candidate_name,
                **agg,
                "best_seed": scored_seed_names[int(np.argmax(scores))] if scores else "",
                "missing_seed_count": len(missing_seeds),
            })

    result = pd.DataFrame(rows)
    result = result.dropna(subset=["score"]).sort_values("score", ascending=False)
    if top_k and top_k > 0:
        result = result.head(top_k)
    return result.reset_index(drop=True), missing_seeds


def save_results(df, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Rank candidate genes by predicted interaction strength to seed genes."
    )
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data", default=os.path.join(project_root, "data", "processed", "graph.pt"))
    parser.add_argument("--output", default=os.path.join(project_root, "results", "gene_rankings.csv"))
    parser.add_argument("--seeds", nargs="*", default=None, help="Seed genes, comma-separated or space-separated.")
    parser.add_argument("--seeds-file", default=None, help="Text file with one seed gene per line.")
    parser.add_argument("--candidates", nargs="*", default=None, help="Optional candidate genes to rank.")
    parser.add_argument("--candidates-file", default=None, help="Text file with one candidate gene per line.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top candidates to save; <=0 saves all.")
    parser.add_argument("--aggregate", choices=["mean", "max", "min"], default="mean")
    parser.add_argument("--include-known", action="store_true", help="Also score already observed seed-candidate links.")
    parser.add_argument("--encoder", choices=["gcn", "gat", "sage"], default="sage")
    parser.add_argument("--decoder", choices=["dot", "mlp", "edge_mlp"], default="mlp")
    parser.add_argument("--use-identity", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--neg-ratio", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    data = torch.load(args.data, weights_only=False)

    seed_genes = load_name_list(args.seeds, args.seeds_file) or DEFAULT_SEEDS
    candidate_genes = load_name_list(args.candidates, args.candidates_file)

    edge_attr_dim = int(data.edge_attr.size(1)) if hasattr(data, "edge_attr") else 8
    config = {
        "seed_genes": seed_genes,
        "encoder": args.encoder,
        "decoder": args.decoder,
        "use_identity": args.use_identity,
        "hidden_dim": args.hidden_dim,
        "out_dim": args.out_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "val_fraction": args.val_fraction,
        "neg_ratio": args.neg_ratio,
        "seed": args.seed,
        "device": args.device,
        "edge_attr_dim": edge_attr_dim,
        "verbose": not args.quiet,
        "log_interval": 50,
    }

    bundle = train_ranking_model(data, config)
    rankings, missing_seeds = rank_candidates(
        bundle,
        seed_genes=seed_genes,
        candidate_genes=candidate_genes,
        top_k=args.top_k,
        batch_size=args.batch_size,
        aggregate=args.aggregate,
        exclude_known=not args.include_known,
        device=args.device,
    )
    save_results(rankings, args.output)

    print(f"Saved rankings to {args.output}")
    print(f"Best validation AUC: {bundle['best_val_auc']:.4f}")
    if missing_seeds:
        print(f"Missing seed genes ignored: {', '.join(missing_seeds)}")
    if not rankings.empty:
        print(rankings.to_string(index=False))


if __name__ == "__main__":
    main()
