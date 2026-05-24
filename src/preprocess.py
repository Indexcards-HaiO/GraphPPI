#!/usr/bin/env python3
"""
GraphPPI 数据预处理脚本
- 从 raw edges.tsv 提取 8 通道边特征 (edge_attr)
- 生成包含 edge_attr 的 graph.pt
- 不固化节点特征（改为训练时动态计算，消除数据泄露）
"""

import pandas as pd
import torch
from torch_geometric.data import Data
import os


def load_edges_with_attrs(edges_filepath):
    """
    读取边列表，构建无向图，同时提取 8 通道边特征
    
    参数:
        edges_filepath: tsv文件路径
    
    返回:
        edge_index: tensor (2, num_undirected_edges)
        edge_weight: tensor (num_undirected_edges,) — combined_score
        edge_attr: tensor (num_undirected_edges, 8) — 8 个证据通道
        node_list: list of unique node names
        node_to_idx: dict mapping node name to index
    """
    df = pd.read_csv(edges_filepath, sep='\t')
    df.columns = df.columns.str.strip('#')

    # 8 个证据通道
    evidence_cols = [
        'neighborhood_on_chromosome',
        'gene_fusion',
        'phylogenetic_cooccurrence',
        'homology',
        'coexpression',
        'experimentally_determined_interaction',
        'database_annotated',
        'automated_textmining'
    ]

    # 转为无向：对每对 (u, v) 保留 combined_score 最高的那条
    edges_dict = {}
    for _, row in df.iterrows():
        u, v = row['node1'], row['node2']
        score = row['combined_score']
        key = tuple(sorted([u, v]))
        if key not in edges_dict or edges_dict[key]['combined_score'] < score:
            evidence = row[evidence_cols].values.astype(float)
            edges_dict[key] = {
                'combined_score': score,
                'evidence': evidence
            }

    all_nodes = sorted(set([node for pair in edges_dict.keys() for node in pair]))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    edge_index_list = []
    edge_weight_list = []
    edge_attr_list = []

    for (u, v), info in edges_dict.items():
        edge_index_list.append([node_to_idx[u], node_to_idx[v]])
        edge_weight_list.append(info['combined_score'])
        edge_attr_list.append(info['evidence'])

    # 添加反向边（无向图需要双向边）
    edge_index_list_rev = [[v, u] for [u, v] in edge_index_list]
    all_edges = edge_index_list + edge_index_list_rev
    all_weights = edge_weight_list + edge_weight_list
    all_attrs = edge_attr_list + edge_attr_list

    edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(all_weights, dtype=torch.float)
    edge_attr = torch.tensor(all_attrs, dtype=torch.float)

    return edge_index, edge_weight, edge_attr, all_nodes, node_to_idx


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    edges_path = os.path.join(raw_dir, 'edges.tsv')

    print("=" * 60)
    print("GraphPPI 数据预处理")
    print("=" * 60)

    # 加载数据
    print("\n读取 edges.tsv 并提取 8 通道边特征...")
    edge_index, edge_weight, edge_attr, node_list, node_to_idx = load_edges_with_attrs(edges_path)

    print(f"  节点数: {len(node_list)}")
    print(f"  无向边数: {edge_index.size(1) // 2}")
    print(f"  有向边对（用于GNN）: {edge_index.size(1)}")
    print(f"  边特征维度: {edge_attr.shape}")
    print(f"    - 8 个证据通道: neighborhood, gene_fusion, phylogenetic_cooccurrence, ")
    print(f"      homology, coexpression, experimentally_determined, database_annotated, textmining")
    print(f"  combined_score 范围: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")
    print(f"  边特征各通道统计:")
    evidence_names = [
        'neighborhood', 'gene_fusion', 'phylo_cooccur', 'homology',
        'coexpression', 'experimental', 'database', 'textmining'
    ]
    for i, name in enumerate(evidence_names):
        nonzero = (edge_attr[:, i] > 0).sum().item()
        print(f"    {name:20s}: non-zero={nonzero:5d}/{edge_attr.size(0):5d}")

    # 构建 Data 对象（不包含节点特征 x，训练时动态计算）
    data = Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_attr=edge_attr,
        num_nodes=len(node_list)
    )

    # 保存节点名称列表（用于种子基因特征计算）
    data.node_names = node_list

    output_path = os.path.join(processed_dir, 'graph.pt')
    torch.save(data, output_path)
    print(f"\n已保存到: {output_path}")
    print(f"  节点数: {data.num_nodes}")
    print(f"  边数（有向）: {data.num_edges}")
    print(f"  edge_attr: {list(data.edge_attr.shape)}")
    print(f"  edge_weight: {list(data.edge_weight.shape)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
