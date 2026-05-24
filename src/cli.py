#!/usr/bin/env python3
"""GraphPPI 命令行入口"""

import argparse
import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'graph.pt')


def cmd_preprocess(args):
    """提取 STRING 8通道边特征"""
    from graphppi.preprocess import main
    main()


def cmd_evaluate(args):
    """GNN k-fold CV 评估"""
    from graphppi.evaluate_gnn import run_full_evaluation
    data = torch.load(DATA_PATH, weights_only=False)
    run_full_evaluation(data, k=args.k)


def cmd_baselines(args):
    """Baseline k-fold CV 评估"""
    from graphppi.evaluate_baselines import evaluate_baselines_kfold
    data = torch.load(DATA_PATH, weights_only=False)
    evaluate_baselines_kfold(data, k=args.k, seed=args.seed)


def cmd_ablation(args):
    """消融实验"""
    from graphppi.ablation import run_feature_ablation, run_architecture_ablation, print_results_table
    data = torch.load(DATA_PATH, weights_only=False)
    r1 = run_feature_ablation(data, k=args.k)
    r2 = run_architecture_ablation(data, k=args.k)
    print_results_table(r1, 'Feature & Decoder Ablation')
    print_results_table(r2, 'Architecture Ablation')


def cmd_rank(args):
    """候选基因/节点排序"""
    from graphppi.rank_genes import (
        DEFAULT_SEEDS,
        load_name_list,
        rank_candidates,
        save_results,
        train_ranking_model,
    )

    data = torch.load(DATA_PATH, weights_only=False)
    seed_genes = load_name_list(args.seeds, args.seeds_file) or DEFAULT_SEEDS
    candidate_genes = load_name_list(args.candidates, args.candidates_file)
    edge_attr_dim = int(data.edge_attr.size(1)) if hasattr(data, "edge_attr") else 8

    bundle = train_ranking_model(data, {
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
    })
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


def main():
    parser = argparse.ArgumentParser(
        description="GraphPPI: Graph Neural Network for PPI Link Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  graphppi preprocess                     # Extract STRING edge features
  graphppi evaluate --k 5                 # 5-fold GNN CV
  graphppi baselines --k 5                # 5-fold baseline CV
  graphppi ablation --k 3                 # Ablation study
  graphppi rank --seeds TP53 BRCA1        # Rank candidate genes
"""
    )
    sub = parser.add_subparsers(dest="command")

    p1 = sub.add_parser("preprocess", help="Extract 8-channel STRING edge features")
    p1.set_defaults(func=cmd_preprocess)

    p2 = sub.add_parser("evaluate", help="Run GNN k-fold cross-validation")
    p2.add_argument("--k", type=int, default=5, help="Number of folds")
    p2.set_defaults(func=cmd_evaluate)

    p3 = sub.add_parser("baselines", help="Run baseline k-fold cross-validation")
    p3.add_argument("--k", type=int, default=5, help="Number of folds")
    p3.add_argument("--seed", type=int, default=42)
    p3.set_defaults(func=cmd_baselines)

    p4 = sub.add_parser("ablation", help="Run ablation study")
    p4.add_argument("--k", type=int, default=3, help="Number of folds")
    p4.set_defaults(func=cmd_ablation)

    p5 = sub.add_parser("rank", help="Rank candidate genes/nodes against seed genes")
    p5.add_argument("--output", default=os.path.join(PROJECT_ROOT, "results", "gene_rankings.csv"))
    p5.add_argument("--seeds", nargs="*", default=None, help="Seed genes, comma-separated or space-separated")
    p5.add_argument("--seeds-file", default=None, help="Text file with one seed gene per line")
    p5.add_argument("--candidates", nargs="*", default=None, help="Optional candidate genes to rank")
    p5.add_argument("--candidates-file", default=None, help="Text file with one candidate gene per line")
    p5.add_argument("--top-k", type=int, default=20, help="Number of rows to save; <=0 saves all")
    p5.add_argument("--aggregate", choices=["mean", "max", "min"], default="mean")
    p5.add_argument("--include-known", action="store_true", help="Also score observed seed-candidate links")
    p5.add_argument("--encoder", choices=["gcn", "gat", "sage"], default="sage")
    p5.add_argument("--decoder", choices=["dot", "mlp", "edge_mlp"], default="mlp")
    p5.add_argument("--use-identity", action="store_true")
    p5.add_argument("--hidden-dim", type=int, default=128)
    p5.add_argument("--out-dim", type=int, default=64)
    p5.add_argument("--num-layers", type=int, default=2)
    p5.add_argument("--dropout", type=float, default=0.5)
    p5.add_argument("--lr", type=float, default=0.005)
    p5.add_argument("--weight-decay", type=float, default=1e-4)
    p5.add_argument("--epochs", type=int, default=300)
    p5.add_argument("--patience", type=int, default=30)
    p5.add_argument("--val-fraction", type=float, default=0.15)
    p5.add_argument("--neg-ratio", type=int, default=1)
    p5.add_argument("--batch-size", type=int, default=8192)
    p5.add_argument("--seed", type=int, default=42)
    p5.add_argument("--device", default="cpu")
    p5.add_argument("--quiet", action="store_true")
    p5.set_defaults(func=cmd_rank)

    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
