# src/data_loader.py

import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import numpy as np
import networkx as nx
from features import compute_all_node_features, compute_degree, compute_weighted_degree, compute_clustering_coefficient, compute_neighbor_weight_mean, compute_seed_neighbor_count


def load_edges(edges_filepath, keep_directed=False):
    """
    读取边列表，构建edge_index和edge_weight
    
    参数:
        edges_filepath: tsv文件路径
        keep_directed: 是否保留有向边（默认False，转为无向）
    
    返回:
        edge_index: tensor (2, num_undirected_edges)
        edge_weight: tensor (num_undirected_edges,)
        node_list: list of unique node names
        node_to_idx: dict mapping node name to index
    """
    df = pd.read_csv(edges_filepath, sep='\t')
    df.columns = df.columns.str.strip('#')
    
    if keep_directed:
        node1_list = df['node1'].tolist()
        node2_list = df['node2'].tolist()
        weights = df['combined_score'].tolist()
        
        all_nodes = list(set(node1_list + node2_list))
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        edge_index = [[node_to_idx[node1_list[i]], node_to_idx[node2_list[i]]] for i in range(len(node1_list))]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights, dtype=torch.float)
        
        return edge_index, edge_weight, all_nodes, node_to_idx
    
    else:
        edges_dict = {}
        for _, row in df.iterrows():
            u, v = row['node1'], row['node2']
            score = row['combined_score']
            key = tuple(sorted([u, v]))
            if key not in edges_dict or edges_dict[key] < score:
                edges_dict[key] = score
        
        all_nodes = list(set([node for pair in edges_dict.keys() for node in pair]))
        node_to_idx = {node: i for i, node in enumerate(all_nodes)}
        
        edge_index = []
        edge_weight = []
        for (u, v), score in edges_dict.items():
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_weight.append(score)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)
        
        return edge_index, edge_weight, all_nodes, node_to_idx


def load_node_degrees(degrees_filepath, node_to_idx):
    """
    读取节点度文件，构建节点度特征张量
    
    参数:
        degrees_filepath: tsv文件路径
        node_to_idx: 节点名称到索引的映射
    
    返回:
        degree_features: tensor (num_nodes, 1)
    """
    df = pd.read_csv(degrees_filepath, sep='\t')
    df.columns = df.columns.str.strip('#')
    
    num_nodes = len(node_to_idx)
    degree_features = torch.zeros(num_nodes, 1)
    
    for _, row in df.iterrows():
        node = row['node']
        degree = row['node_degree']
        if node in node_to_idx:
            idx = node_to_idx[node]
            degree_features[idx, 0] = degree
    
    return degree_features


def load_node_annotations(annotations_filepath, node_to_idx):
    """
    读取注释文件，构建节点注释特征（当前返回零向量占位符）
    
    参数:
        annotations_filepath: tsv文件路径
        node_to_idx: 节点名称到索引的映射
    
    返回:
        annotation_features: tensor (num_nodes, 1)
    """
    df = pd.read_csv(annotations_filepath, sep='\t')
    df.columns = df.columns.str.strip('#')
    
    num_nodes = len(node_to_idx)
    annotation_features = torch.zeros(num_nodes, 1)
    
    return annotation_features


def compute_clustering_coefficient(edge_index, num_nodes):
    """
    计算每个节点的聚类系数
    
    参数:
        edge_index: tensor (2, num_edges)
        num_nodes: int
    
    返回:
        clustering_coeff: tensor (num_nodes, 1)
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    clustering_dict = nx.clustering(G)
    clustering_coeff = torch.zeros(num_nodes, 1)
    
    for node, coeff in clustering_dict.items():
        clustering_coeff[node, 0] = coeff
    
    return clustering_coeff


def compute_eigenvector_centrality(edge_index, num_nodes, max_iter=100, tol=1e-6):
    """
    计算每个节点的特征向量中心性
    
    参数:
        edge_index: tensor (2, num_edges)
        num_nodes: int
        max_iter: 最大迭代次数
        tol: 收敛容忍度
    
    返回:
        eigenvector_centrality: tensor (num_nodes, 1)
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    try:
        eigenvector_dict = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol)
        eigenvector_centrality = torch.zeros(num_nodes, 1)
        for node, centrality in eigenvector_dict.items():
            eigenvector_centrality[node, 0] = centrality
    except nx.PowerIterationFailedConvergence:
        print("警告: 特征向量中心性幂迭代未收敛，使用零值替代")
        eigenvector_centrality = torch.zeros(num_nodes, 1)
    
    return eigenvector_centrality


def compute_average_neighbor_weight(edge_index, edge_weight, num_nodes):
    """
    计算每个节点的平均邻居边权重
    
    参数:
        edge_index: tensor (2, num_edges)
        edge_weight: tensor (num_edges,)
        num_nodes: int
    
    返回:
        avg_neighbor_weight: tensor (num_nodes, 1)
    """
    avg_neighbor_weight = torch.zeros(num_nodes, 1)
    
    for i in range(num_nodes):
        neighbors_mask = (edge_index[0] == i) | (edge_index[1] == i)
        if not neighbors_mask.any():
            avg_neighbor_weight[i, 0] = 0.0
            continue
        
        neighbor_weights = edge_weight[neighbors_mask]
        avg_neighbor_weight[i, 0] = neighbor_weights.mean().item()
    
    return avg_neighbor_weight


def compute_all_node_features(edge_index, edge_weight, num_nodes, node_names):
    """
    计算所有5个节点特征，返回标准化后的特征矩阵
    
    参数:
        edge_index: tensor (2, num_edges)
        edge_weight: tensor (num_edges,)
        num_nodes: int
        node_names: list of node names
    
    返回:
        x: tensor (num_nodes, 5)
        feature_names: list of feature names
        feature_dict: dict of raw features (for debugging)
    """
    print("计算节点特征...")
    
    # 1. degree
    print("  计算 degree...")
    degree = compute_degree(edge_index, num_nodes)
    
    # 2. weighted degree
    print("  计算 weighted degree...")
    weighted_degree = compute_weighted_degree(edge_index, edge_weight, num_nodes)
    
    # 3. clustering coefficient
    print("  计算 clustering coefficient...")
    clustering = compute_clustering_coefficient(edge_index, num_nodes)
    
    # 4. neighbor weight mean
    print("  计算 neighbor weight mean...")
    neighbor_weight_mean = compute_neighbor_weight_mean(edge_index, edge_weight, num_nodes)
    
    # 5. seed-neighbor count
    print("  计算 seed neighbor count...")
    seed_genes = ['TP53', 'BRCA1', 'ERBB2', 'PIK3CA', 'ESR1']
    seed_neighbor_count = compute_seed_neighbor_count(edge_index, num_nodes, seed_genes, node_names)
    
    # 确保每个特征都是 Tensor 类型
    features = [
        degree,
        weighted_degree,
        clustering,
        neighbor_weight_mean,
        seed_neighbor_count
    ]
    
    # 检查并转换每个特征
    for i, f in enumerate(features):
        if not isinstance(f, torch.Tensor):
            features[i] = torch.tensor(f, dtype=torch.float)
        elif f.dim() == 1:
            features[i] = f.reshape(-1, 1)
    
    # 拼接特征
    x = torch.cat(features, dim=1)
    
    feature_names = ['degree', 'weighted_degree', 'clustering_coefficient', 
                     'neighbor_weight_mean', 'seed_neighbor_count']
    
    # 标准化
    scaler = StandardScaler()
    x_numpy = scaler.fit_transform(x.numpy())
    x = torch.tensor(x_numpy, dtype=torch.float)
    
    print(f"  特征矩阵形状: {x.shape}")
    print(f"  特征名称: {feature_names}")
    
    return x, feature_names, None


def build_graph(edges_filepath, degrees_filepath, annotations_filepath, use_all_features=True):
    """
    构建完整的PyG Data对象
    """
    edge_index, edge_weight, node_list, node_to_idx = load_edges(edges_filepath, keep_directed=False)
    _ = load_node_degrees(degrees_filepath, node_to_idx)  # 不再使用STRING的node_degree
    _ = load_node_annotations(annotations_filepath, node_to_idx)
    
    if not use_all_features:
        # 使用单位矩阵（无特征基线）
        x = torch.eye(len(node_list))
        feature_names = ['identity'] * len(node_list)
        print(f"使用单位矩阵特征，维度: {x.shape[1]}")
    else:
        # 使用5个拓扑特征
        x, feature_names, _ = compute_all_node_features(edge_index, edge_weight, len(node_list), node_list)
    
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    
    return data, node_list, feature_names


if __name__ == "__main__":
    edges_path = "/home/indexcards/GraphPPI/data/raw/edges.tsv"
    degrees_path = "/home/indexcards/GraphPPI/data/raw/degrees.tsv"
    annotations_path = "/home/indexcards/GraphPPI/data/raw/annotations.tsv"
    
    data, node_list, feature_names = build_graph(edges_path, degrees_path, annotations_path, use_all_features=True)
    
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"特征矩阵形状: {data.x.shape}")
    print(f"特征名称: {feature_names}")
    print(f"边权重范围: {data.edge_weight.min():.3f} - {data.edge_weight.max():.3f}")
    print(f"节点示例: {node_list[:5]}")

    # 保存处理后的数据
    torch.save(data, "/home/indexcards/GraphPPI/data/processed/graph.pt")
    print(f"\n已保存图对象至: data/processed/graph.pt")