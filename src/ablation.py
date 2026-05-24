#!/usr/bin/env python3
"""
消融实验编排
系统化对比不同组件的影响
"""

import torch
import os
import argparse
import pandas as pd

from graphppi.evaluate_gnn import evaluate_gnn_kfold


def run_feature_ablation(data, k=5):
    """特征消融：node features, decoder type, edge features"""
    print("\n" + "=" * 60)
    print("消融实验 A: 特征与解码器消融")
    print("=" * 60)

    base_config = {
        'encoder': 'gcn', 'num_layers': 2,
        'hidden_dim': 128, 'out_dim': 64, 'dropout': 0.5,
        'lr': 0.005, 'weight_decay': 1e-4,
        'epochs': 300, 'patience': 30, 'seed': 42,
    }

    experiments = [
        {
            'name': 'A1_identity_dot',
            'use_identity': True, 'decoder': 'dot',
            'description': 'Identity + Dot',
        },
        {
            'name': 'A2_topology_dot',
            'use_identity': False, 'decoder': 'dot',
            'description': '5 Topology + Dot',
        },
        {
            'name': 'A3_topology_mlp',
            'use_identity': False, 'decoder': 'mlp',
            'description': '5 Topology + MLP',
        },
        {
            'name': 'A4_topology_edge_mlp',
            'use_identity': False, 'decoder': 'edge_mlp',
            'description': '5 Topology + EdgeMLP (8-channel STRING)',
        },
    ]

    results = []
    for exp in experiments:
        name = exp.pop('name')
        desc = exp.pop('description')
        cfg = {**base_config, **exp}
        print(f"\n--- {name}: {desc} ---")
        summary, _ = evaluate_gnn_kfold(data, cfg, k=k, verbose=False)

        row = {
            'Experiment': name,
            'Description': desc,
            'AUC': f"{summary['auc_mean']:.4f}±{summary['auc_std']:.3f}",
            'AP': f"{summary['ap_mean']:.4f}±{summary['ap_std']:.3f}",
            'Hits@3': f"{summary.get('hits@3_mean', 0):.4f}±{summary.get('hits@3_std', 0):.3f}",
            'Hits@10': f"{summary.get('hits@10_mean', 0):.4f}±{summary.get('hits@10_std', 0):.3f}",
            'Hits@20': f"{summary.get('hits@20_mean', 0):.4f}±{summary.get('hits@20_std', 0):.3f}",
            'auc_raw': summary['auc_mean'],
        }
        results.append(row)
        print(f"  AUC: {row['AUC']}, AP: {row['AP']}")

    return results


def run_architecture_ablation(data, k=5):
    """架构消融：encoder, layers, decoder"""
    print("\n" + "=" * 60)
    print("消融实验 B: 架构消融")
    print("=" * 60)

    base_config = {
        'use_identity': False, 'hidden_dim': 128, 'out_dim': 64,
        'dropout': 0.5, 'lr': 0.005, 'weight_decay': 1e-4,
        'epochs': 300, 'patience': 30, 'seed': 42,
    }

    experiments = [
        {
            'name': 'B1_GCN2_dot',
            'encoder': 'gcn', 'num_layers': 2, 'decoder': 'dot',
            'description': 'GCN-2L + Dot',
        },
        {
            'name': 'B2_GCN2_mlp',
            'encoder': 'gcn', 'num_layers': 2, 'decoder': 'mlp',
            'description': 'GCN-2L + MLP',
        },
        {
            'name': 'B3_GCN3_mlp',
            'encoder': 'gcn', 'num_layers': 3, 'decoder': 'mlp',
            'description': 'GCN-3L + MLP',
        },
    ]

    results = []
    for exp in experiments:
        name = exp.pop('name')
        desc = exp.pop('description')
        cfg = {**base_config, **exp}
        print(f"\n--- {name}: {desc} ---")
        summary, _ = evaluate_gnn_kfold(data, cfg, k=k, verbose=False)

        row = {
            'Experiment': name,
            'Description': desc,
            'AUC': f"{summary['auc_mean']:.4f}±{summary['auc_std']:.3f}",
            'AP': f"{summary['ap_mean']:.4f}±{summary['ap_std']:.3f}",
            'Hits@10': f"{summary.get('hits@10_mean', 0):.4f}±{summary.get('hits@10_std', 0):.3f}",
            'auc_raw': summary['auc_mean'],
        }
        results.append(row)
        print(f"  AUC: {row['AUC']}, AP: {row['AP']}")

    return results


def print_results_table(results, title):
    """打印结果表格"""
    if not results:
        return
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    # 按 AUC 排序
    df_sorted = df.sort_values('auc_raw', ascending=False)
    display_cols = [c for c in df_sorted.columns if c != 'auc_raw']
    print(df_sorted[display_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=3, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--ablation', type=str, default='all',
                        choices=['all', 'feature', 'architecture'],
                        help='Which ablation to run')
    args = parser.parse_args()

    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'graph.pt'
    )
    data = torch.load(data_path, weights_only=False)

    all_results = []

    if args.ablation in ('all', 'feature'):
        results_a = run_feature_ablation(data, k=args.k)
        all_results.extend(results_a)
        print_results_table(results_a, '消融实验 A: 特征与解码器')

    if args.ablation in ('all', 'architecture'):
        results_b = run_architecture_ablation(data, k=args.k)
        all_results.extend(results_b)
        print_results_table(results_b, '消融实验 B: 架构')

    # Baseline 对照
    print(f"\n{'='*80}")
    print("  Baseline 对照")
    print(f"{'='*80}")
    print(f"  {'Method':<25s} {'AUC':>10s} {'AP':>10s}")
    print(f"  {'-'*45}")
    baselines = {
        'Adamic-Adar': (0.9054, 0.8779),
        'Common Neighbors': (0.8998, 0.8693),
        'Jaccard': (0.8972, 0.8787),
        'Node2Vec+RF': (0.8875, 0.8838),
    }
    for name, (auc, ap) in baselines.items():
        print(f"  {name:<25s} {auc:>8.4f}    {ap:>8.4f}")

    # 保存到 CSV
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results'
    )
    os.makedirs(output_dir, exist_ok=True)
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'ablation_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n结果已保存到: {csv_path}")
