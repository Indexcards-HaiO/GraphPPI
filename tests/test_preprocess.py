"""测试数据加载和预处理"""
import os
import torch
from graphppi.preprocess import load_edges_with_attrs


def test_load_graph():
    """测试 graph.pt 加载"""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'processed', 'graph.pt'
    )
    data = torch.load(data_path, weights_only=False)
    assert data.num_nodes == 146
    assert data.num_edges == 6824
    assert data.edge_attr.shape == (6824, 8)
    assert data.edge_weight.shape == (6824,)
    assert hasattr(data, 'node_names')
    assert len(data.node_names) == 146


def test_load_edges_with_attrs():
    """测试边特征提取"""
    edges_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'raw', 'edges.tsv'
    )
    edge_index, edge_weight, edge_attr, node_list, node_to_idx = \
        load_edges_with_attrs(edges_path)

    assert edge_index.size(0) == 2
    assert edge_attr.size(1) == 8
    assert edge_weight.size(0) == edge_index.size(1)
    assert len(node_list) > 0
    assert len(node_to_idx) == len(node_list)
    # 验证有向边对
    assert edge_index.size(1) % 2 == 0
