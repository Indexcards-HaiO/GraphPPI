#!/usr/bin/env python3
"""
GraphPPI 工具函数
- k-fold 边集划分（无数据泄露）
- 负采样（随机 + hard negative）
- 动态节点特征计算
"""

import torch
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# ============================================================
# 边集划分
# ============================================================

def split_edges_kfold(edge_index, num_edges_undirected, k=5, seed=42):
    """
    对无向边进行 k-fold 划分
    每个 fold 返回 (train_pos_indices, val_pos_indices, test_pos_indices)
    
    参数:
        edge_index: tensor (2, 2*E) — 有向边对
        num_edges_undirected: int — 无向边数 E
        k: fold 数
        seed: 随机种子
    
    返回:
        folds: list of dict, 每个含 'train', 'val', 'test' 边索引
    """
    # 只对无向边做划分（用前 E 条，因为每条边存了两次）
    undirected_indices = np.arange(num_edges_undirected)
    np.random.seed(seed)
    np.random.shuffle(undirected_indices)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []

    # 对每条无向边，用两个有向索引
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(undirected_indices)):
        tv_indices = undirected_indices[train_val_idx]
        test_indices = undirected_indices[test_idx]

        # 从 train_val 中再分出一部分作为验证集（~20% of remaining）
        np.random.seed(seed + fold_idx)
        np.random.shuffle(tv_indices)
        val_size = max(1, int(len(tv_indices) * 0.2))
        val_indices = tv_indices[:val_size]
        train_indices = tv_indices[val_size:]

        folds.append({
            'train': train_indices,
            'val': val_indices,
            'test': test_indices
        })

    return folds


def get_edge_set_from_indices(edge_index, undirected_indices):
    """
    从无向边索引获取有向边对索引
    每条无向边对应两条有向边：idx 和 idx + E
    """
    num_undirected = edge_index.size(1) // 2
    directed_indices = []
    for i in undirected_indices:
        directed_indices.append(int(i))
        directed_indices.append(int(i) + num_undirected)
    return directed_indices


def get_edges_by_indices(edge_index, undirected_indices, num_undirected):
    """
    从无向边索引获取对应的有向边
    """
    directed = get_edge_set_from_indices(edge_index, undirected_indices)
    return edge_index[:, directed]


# ============================================================
# 负采样
# ============================================================

def sample_negative_edges(pos_edges, num_nodes, num_samples, exclude_set=None, random_state=None):
    """
    随机负采样：从未出现的边对中采样
    
    参数:
        pos_edges: tensor (2, num_pos) — 正样本边
        num_nodes: int
        num_samples: 需要采样的负样本数
        exclude_set: set of tuples, 应排除的边
        random_state: 随机种子
    
    返回:
        neg_edges: tensor (2, num_samples)
    """
    if random_state is not None:
        np.random.seed(random_state)

    if exclude_set is None:
        exclude_set = set()

    # 将正样本也加入排除集
    for i in range(pos_edges.size(1)):
        u = int(pos_edges[0, i].item())
        v = int(pos_edges[1, i].item())
        exclude_set.add((u, v))
        exclude_set.add((v, u))

    neg_edges = []
    max_attempts = num_samples * 50
    attempts = 0

    while len(neg_edges) < num_samples and attempts < max_attempts:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        attempts += 1
        if u != v and (u, v) not in exclude_set:
            neg_edges.append([u, v])
            exclude_set.add((u, v))
            exclude_set.add((v, u))

    if len(neg_edges) < num_samples:
        raise RuntimeError(f"只能采样 {len(neg_edges)} 个负样本，需要 {num_samples} 个")

    return torch.tensor(neg_edges, dtype=torch.long).t()


def sample_hard_negative_edges(pos_edges, num_nodes, num_samples, exclude_set, degree):
    """
    度数匹配的难负采样
    """
    if exclude_set is None:
        exclude_set = set()

    # 将正样本加入排除集
    for i in range(pos_edges.size(1)):
        u = int(pos_edges[0, i].item())
        v = int(pos_edges[1, i].item())
        exclude_set.add((u, v))
        exclude_set.add((v, u))

    pos_nodes = torch.unique(pos_edges).tolist()
    degree_np = degree.numpy() if isinstance(degree, torch.Tensor) else np.array(degree)
    pos_degrees = [degree_np[node] for node in pos_nodes]

    neg_edges = []
    exclude = exclude_set.copy()
    max_attempts = num_samples * 100
    attempts = 0

    while len(neg_edges) < num_samples and attempts < max_attempts:
        target_degree = np.random.choice(pos_degrees)
        candidates = [i for i in range(num_nodes) if abs(degree_np[i] - target_degree) < 5]
        if len(candidates) < 2:
            candidates = list(range(num_nodes))

        u = np.random.choice(candidates)
        v = np.random.choice(candidates)
        attempts += 1

        if u != v and (u, v) not in exclude:
            neg_edges.append([u, v])
            exclude.add((u, v))
            exclude.add((v, u))

    if len(neg_edges) < num_samples:
        # 回退到随机采样
        return sample_negative_edges(pos_edges, num_nodes, num_samples, exclude_set)

    return torch.tensor(neg_edges, dtype=torch.long).t()


# ============================================================
# 动态节点特征计算（基于给定边集，消除数据泄露）
# ============================================================

def compute_degree(edge_index, num_nodes):
    """基于给定边集计算节点度"""
    degree = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        degree[edge_index[0, i].item()] += 1
    return degree


def compute_weighted_degree(edge_index, edge_weight, num_nodes):
    """基于给定边集和边权重计算加权度"""
    wdegree = torch.zeros(num_nodes)
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        wdegree[u] += edge_weight[i].item()
    return wdegree


def compute_clustering_coefficient(edge_index, num_nodes):
    """基于给定边集计算聚类系数"""
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    clustering_dict = nx.clustering(G)
    clustering = torch.zeros(num_nodes)
    for node, coeff in clustering_dict.items():
        clustering[node] = coeff
    return clustering


def compute_neighbor_weight_mean(edge_index, edge_weight, num_nodes):
    """基于给定边集计算邻居边权重均值"""
    # 收集每个节点的邻居权重
    neighbor_weights = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        w = edge_weight[i].item()
        neighbor_weights[u].append(w)
        neighbor_weights[v].append(w)

    result = torch.zeros(num_nodes)
    for node in range(num_nodes):
        if neighbor_weights[node]:
            result[node] = np.mean(neighbor_weights[node])
    return result


def compute_seed_neighbor_count(edge_index, num_nodes, node_names, seed_genes):
    """基于给定边集计算与种子基因的邻居重叠数"""
    # 构建邻居集合
    neighbors = {i: set() for i in range(num_nodes)}
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        neighbors[u].add(v)
        neighbors[v].add(u)

    seed_indices = []
    for seed in seed_genes:
        if seed in node_names:
            seed_indices.append(node_names.index(seed))

    result = torch.zeros(num_nodes)
    for node in range(num_nodes):
        count = sum(1 for s in seed_indices if s in neighbors[node])
        result[node] = count
    return result


def compute_features_from_edges(edge_index, edge_weight, num_nodes, node_names):
    """
    基于给定的边集（应该是训练边）动态计算 5 个拓扑特征
    这是消除数据泄露的关键：只用训练边计算特征
    """
    seed_genes = ['TP53', 'BRCA1', 'ERBB2', 'PIK3CA', 'ESR1']

    f_degree = compute_degree(edge_index, num_nodes)
    f_wdegree = compute_weighted_degree(edge_index, edge_weight, num_nodes)
    f_clustering = compute_clustering_coefficient(edge_index, num_nodes)
    f_nbr_weight = compute_neighbor_weight_mean(edge_index, edge_weight, num_nodes)
    f_seed_nbr = compute_seed_neighbor_count(edge_index, num_nodes, node_names, seed_genes)

    # 拼接
    features = torch.stack([f_degree, f_wdegree, f_clustering, f_nbr_weight, f_seed_nbr], dim=1)

    # 标准化
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(features.numpy()), dtype=torch.float)

    return x


# ============================================================
# 数据准备（单个 fold）
# ============================================================

def prepare_fold_data(data, fold_indices, neg_ratio=1, hard_negative=False):
    """
    为一个 fold 准备训练/验证/测试数据
    
    参数:
        data: PyG Data 对象（全图）
        fold_indices: dict with 'train', 'val', 'test' 无向边索引
        neg_ratio: 负样本比例（neg/pos）
        hard_negative: 是否使用难负采样
    
    返回:
        train_data: dict
        val_data: dict
        test_data: dict
        node_features: tensor (num_nodes, 5)
    """
    num_nodes = data.num_nodes
    num_undirected = data.edge_index.size(1) // 2
    node_names = data.node_names

    # 获取各集合的有向边
    train_pos = get_edges_by_indices(data.edge_index, fold_indices['train'], num_undirected)
    val_pos = get_edges_by_indices(data.edge_index, fold_indices['val'], num_undirected)
    test_pos = get_edges_by_indices(data.edge_index, fold_indices['test'], num_undirected)

    # 只取无向（去重后的）边用于消息传递图
    # train_pos 的前半部分是无向边（后半是反向）
    train_undirected_pos = train_pos[:, :train_pos.size(1) // 2]

    # 消息传递图：只使用训练正边（无向）
    mp_edge_index = train_undirected_pos

    # 消息传递图的边权重（必须与 mp_edge_index 的边数匹配）
    train_undir_indices = fold_indices['train']
    mp_edge_weight = data.edge_weight[train_undir_indices]  # 只取训练边的权重

    # 动态计算节点特征（只基于训练边）
    # 构建训练边的有向版本用于特征计算
    num_train_undir = len(train_undir_indices)
    train_edge_index_for_features = torch.zeros(2, 2 * num_train_undir, dtype=torch.long)
    train_edge_weight_for_features = torch.zeros(2 * num_train_undir)
    for i, idx in enumerate(train_undir_indices):
        u = data.edge_index[0, idx].item()
        v = data.edge_index[1, idx].item()
        w = data.edge_weight[idx].item()
        train_edge_index_for_features[0, i] = u
        train_edge_index_for_features[1, i] = v
        train_edge_weight_for_features[i] = w
        # 反向边
        train_edge_index_for_features[0, i + num_train_undir] = v
        train_edge_index_for_features[1, i + num_train_undir] = u
        train_edge_weight_for_features[i + num_train_undir] = w

    x = compute_features_from_edges(
        train_edge_index_for_features, train_edge_weight_for_features,
        num_nodes, node_names
    )

    # 构建排除集合（所有训练正边 + val/test 正边）
    exclude_set = set()
    for edges in [train_pos, val_pos, test_pos]:
        for i in range(edges.size(1)):
            u = int(edges[0, i].item())
            v = int(edges[1, i].item())
            exclude_set.add((u, v))
            exclude_set.add((v, u))

    # 负采样
    num_train_neg = train_undirected_pos.size(1) * neg_ratio
    num_val_neg = val_pos.size(1) // 2 * neg_ratio  # val_pos 也是双向的
    num_test_neg = test_pos.size(1) // 2 * neg_ratio

    if hard_negative:
        degree = compute_degree(data.edge_index, num_nodes)
        train_neg = sample_hard_negative_edges(train_undirected_pos, num_nodes, num_train_neg, exclude_set, degree)
        val_neg = sample_hard_negative_edges(val_pos, num_nodes, num_val_neg, exclude_set, degree)
        test_neg = sample_hard_negative_edges(test_pos, num_nodes, num_test_neg, exclude_set, degree)
    else:
        train_neg = sample_negative_edges(train_undirected_pos, num_nodes, num_train_neg, exclude_set)
        val_neg = sample_negative_edges(val_pos, num_nodes, num_val_neg, exclude_set)
        test_neg = sample_negative_edges(test_pos, num_nodes, num_test_neg, exclude_set)

    # 组装训练/验证/测试 边和标签
    def build_edge_set(pos, neg):
        edges = torch.cat([pos, neg], dim=1)
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
        return edges, labels

    train_edges_all, train_labels = build_edge_set(train_undirected_pos, train_neg)
    val_edges_all, val_labels = build_edge_set(val_pos[:, :val_pos.size(1)//2], val_neg)  # 取无向部分
    test_edges_all, test_labels = build_edge_set(test_pos[:, :test_pos.size(1)//2], test_neg)

    # 边特征：使用哈希表快速查找（O(1) 而非 O(N*M)）
    # 构建 (u, v) -> attr 的映射
    _edge_attr_map = {}
    for j in range(data.edge_index.size(1)):
        u = int(data.edge_index[0, j].item())
        v = int(data.edge_index[1, j].item())
        _edge_attr_map[(u, v)] = data.edge_attr[j]

    def get_edge_attr_fast(edge_pairs):
        """快速获取边特征，正样本从全图取，负样本用 0"""
        attr = torch.zeros(edge_pairs.size(1), 8)
        for i in range(edge_pairs.size(1)):
            key = (int(edge_pairs[0, i].item()), int(edge_pairs[1, i].item()))
            if key in _edge_attr_map:
                attr[i] = _edge_attr_map[key]
        return attr

    train_edge_attr = get_edge_attr_fast(train_edges_all)
    val_edge_attr = get_edge_attr_fast(val_edges_all)
    test_edge_attr = get_edge_attr_fast(test_edges_all)

    return {
        'x': x,
        'mp_edge_index': mp_edge_index,
        'mp_edge_weight': mp_edge_weight,
        'train_edges': train_edges_all,
        'train_labels': train_labels,
        'train_edge_attr': train_edge_attr,
        'val_edges': val_edges_all,
        'val_labels': val_labels,
        'val_edge_attr': val_edge_attr,
        'test_edges': test_edges_all,
        'test_labels': test_labels,
        'test_edge_attr': test_edge_attr,
    }
