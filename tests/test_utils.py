"""测试 k-fold 划分、负采样、特征计算"""
import torch
import numpy as np
from graphppi.utils import (
    split_edges_kfold, sample_negative_edges,
    compute_degree, compute_features_from_edges,
)


def test_split_edges_kfold():
    """k-fold 划分基本测试"""
    # 简单图: 4节点, 3条无向边 → 6条有向边
    edge_index = torch.tensor([
        [0, 1, 2, 1, 2, 3],  # 前半: 无向, 后半: 反向
        [1, 2, 3, 0, 1, 2],
    ], dtype=torch.long)
    num_undirected = 3

    folds = split_edges_kfold(edge_index, num_undirected, k=2, seed=42)
    assert len(folds) == 2

    for fold in folds:
        # 所有边恰好分配一次
        all_indices = set(fold['train']) | set(fold['test'])
        assert all_indices == set(range(3))
        # train 和 test 不重叠
        assert set(fold['train']).isdisjoint(set(fold['test']))


def test_sample_negative_edges():
    """负采样基本测试"""
    pos_edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    num_nodes = 10
    num_samples = 5

    neg = sample_negative_edges(pos_edges, num_nodes, num_samples)
    assert neg.size(1) == num_samples
    # 负样本不应与正样本重复
    for i in range(neg.size(1)):
        pair = (int(neg[0, i].item()), int(neg[1, i].item()))
        assert pair != (0, 1) and pair != (1, 0)


def test_compute_degree():
    """度计算测试"""
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    num_nodes = 3
    degree = compute_degree(edge_index, num_nodes)
    assert degree[0] == 2  # 连 1 和 2
    assert degree[1] == 2  # 连 0 和 2
    assert degree[2] == 2  # 连 0 和 1


def test_compute_features_from_edges():
    """特征计算基本测试"""
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([0.5, 0.5])
    node_names = ['A', 'B']
    x = compute_features_from_edges(edge_index, edge_weight, 2, node_names)
    assert x.shape == (2, 5)  # 5维拓扑特征
    assert not torch.isnan(x).any()
