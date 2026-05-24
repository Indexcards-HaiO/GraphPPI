# src/baselines/adamic_adar.py

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def adamic_adar_predict(edge_index, num_nodes, test_edges, test_labels):
    """
    使用Adamic-Adar指数进行链接预测
    
    参数:
        edge_index: tensor (2, num_edges) 训练集边
        num_nodes: int
        test_edges: tensor (2, num_test_edges) 测试边
        test_labels: tensor (num_test_edges,) 真实标签
    
    返回:
        auc: float
        ap: float
        scores: array 预测分数
    """
    # 构建邻接矩阵和节点度
    adj = {}
    degree = {}
    
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    for node, neighbors in adj.items():
        degree[node] = len(neighbors)
    
    # 计算每条测试边的Adamic-Adar指数
    scores = []
    for i in range(test_edges.size(1)):
        u = test_edges[0, i].item()
        v = test_edges[1, i].item()
        
        neighbors_u = adj.get(u, set())
        neighbors_v = adj.get(v, set())
        
        common = neighbors_u & neighbors_v
        
        aa_score = 0.0
        for w in common:
            d_w = degree.get(w, 0)
            if d_w > 1:
                aa_score += 1.0 / np.log(d_w)
        
        scores.append(aa_score)
    
    scores = np.array(scores)
    auc = roc_auc_score(test_labels.numpy(), scores)
    ap = average_precision_score(test_labels.numpy(), scores)
    
    return auc, ap, scores