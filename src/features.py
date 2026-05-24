# src/features.py

import torch
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler


def compute_degree(edge_index, num_nodes):
    """
    计算节点度
    
    参数:
        edge_index: tensor (2, num_edges)
        num_nodes: int
    
    返回:
        degree: tensor (num_nodes, 1)
    """
    degree = torch.zeros(num_nodes, 1)
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        degree[u, 0] += 1
        degree[v, 0] += 1
    return degree


def compute_weighted_degree(edge_index, edge_weight, num_nodes):
    """
    计算加权度（节点邻居边权重之和）
    
    参数:
        edge_index: tensor (2, num_edges)
        edge_weight: tensor (num_edges,)
        num_nodes: int
    
    返回:
        weighted_degree: tensor (num_nodes, 1)
    """
    weighted_degree = torch.zeros(num_nodes, 1)
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        w = edge_weight[i].item()
        weighted_degree[u, 0] += w
        weighted_degree[v, 0] += w
    return weighted_degree


def compute_clustering_coefficient(edge_index, num_nodes):
    """
    计算局部聚类系数
    
    参数:
        edge_index: tensor (2, num_edges)
        num_nodes: int
    
    返回:
        clustering: tensor (num_nodes, 1)
    """
    # 转换为networkx图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # 计算聚类系数
    clustering_dict = nx.clustering(G)
    clustering = torch.zeros(num_nodes, 1)
    for node, coeff in clustering_dict.items():
        clustering[node, 0] = coeff
    
    return clustering


def compute_neighbor_weight_mean(edge_index, edge_weight, num_nodes):
    """
    计算邻居边权重的均值
    
    参数:
        edge_index: tensor (2, num_edges)
        edge_weight: tensor (num_edges,)
        num_nodes: int
    
    返回:
        neighbor_weight_mean: tensor (num_nodes, 1)
    """
    neighbor_weight_mean = torch.zeros(num_nodes, 1)
    
    for i in range(num_nodes):
        # 找到与节点i相连的所有边
        connected_edges = []
        for j in range(edge_index.size(1)):
            u = edge_index[0, j].item()
            v = edge_index[1, j].item()
            if u == i or v == i:
                connected_edges.append(edge_weight[j].item())
        
        if len(connected_edges) > 0:
            neighbor_weight_mean[i, 0] = np.mean(connected_edges)
        else:
            neighbor_weight_mean[i, 0] = 0.0
    
    return neighbor_weight_mean


def compute_seed_neighbor_count(edge_index, num_nodes, seed_genes, node_names):
    """
    计算与种子基因的邻居重叠数
    
    种子基因包括: TP53, BRCA1, ERBB2, PIK3CA, ESR1
    
    参数:
        edge_index: tensor (2, num_edges)
        num_nodes: int
        seed_genes: list of seed gene names
        node_names: list of node names (与节点索引对应)
    
    返回:
        seed_neighbor_count: tensor (num_nodes, 1)
    """
    # 找到种子基因的节点索引
    seed_indices = []
    for seed in seed_genes:
        if seed in node_names:
            seed_indices.append(node_names.index(seed))
    
    # 构建每个节点的邻居集合
    node_neighbors = {}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        node_neighbors.setdefault(u, set()).add(v)
        node_neighbors.setdefault(v, set()).add(u)
    
    # 计算每个节点与种子基因的邻居重叠数
    seed_neighbor_count = torch.zeros(num_nodes, 1)
    
    for i in range(num_nodes):
        neighbors = node_neighbors.get(i, set())
        count = 0
        for seed_idx in seed_indices:
            if seed_idx in neighbors:
                count += 1
        seed_neighbor_count[i, 0] = count
    
    return seed_neighbor_count


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
    
    # 拼接特征
    features = torch.cat([degree, weighted_degree, clustering, 
                          neighbor_weight_mean, seed_neighbor_count], dim=1)
    
    feature_names = ['degree', 'weighted_degree', 'clustering_coefficient', 
                     'neighbor_weight_mean', 'seed_neighbor_count']
    
    # 标准化
    scaler = StandardScaler()
    features_numpy = scaler.fit_transform(features.numpy())
    x = torch.tensor(features_numpy, dtype=torch.float)
    
    print(f"  特征矩阵形状: {x.shape}")
    print(f"  特征名称: {feature_names}")
    
    # 返回原始特征用于调试
    feature_dict = {
        'degree': degree,
        'weighted_degree': weighted_degree,
        'clustering_coefficient': clustering,
        'neighbor_weight_mean': neighbor_weight_mean,
        'seed_neighbor_count': seed_neighbor_count
    }
    
    return x, feature_names, feature_dict


if __name__ == "__main__":
    # 测试特征计算
    import sys
    sys.path.append('.')
    
    # 加载数据
    from src.data_loader import load_edges, load_node_degrees, load_node_annotations
    
    edges_path = "data/raw/edges.tsv"
    degrees_path = "data/raw/degrees.tsv"
    annotations_path = "data/raw/annotations.tsv"
    
    # 加载边和节点
    edge_index, edge_weight, node_list, node_to_idx = load_edges(edges_path, keep_directed=False)
    
    print(f"节点数: {len(node_list)}")
    print(f"边数: {edge_index.size(1)}")
    print()
    
    # 计算特征
    x, feature_names, feature_dict = compute_all_node_features(edge_index, edge_weight, len(node_list), node_list)
    
    print("\n特征统计:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: mean={x[:, i].mean():.4f}, std={x[:, i].std():.4f}")