# src/train_link_prediction.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import argparse


class GCNLinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return torch.sigmoid((z_u * z_v).sum(dim=-1))


def sample_negative_edges(pos_edges, num_nodes, num_samples, exclude_set=None):
    """从非边中采样负样本"""
    if exclude_set is None:
        exclude_set = set()
    
    neg_edges = []
    exclude = exclude_set.copy()
    
    for i in range(pos_edges.size(1)):
        u = pos_edges[0, i].item()
        v = pos_edges[1, i].item()
        exclude.add((u, v))
        exclude.add((v, u))
    
    while len(neg_edges) < num_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (u, v) not in exclude and (v, u) not in exclude:
            neg_edges.append([u, v])
            exclude.add((u, v))
            exclude.add((v, u))
    
    return torch.tensor(neg_edges, dtype=torch.long).t()


def train_gcn_link_prediction(data, config_name='identity', use_edge_weight=False, 
                               test_size=0.2, val_size=0.2, 
                               epochs=500, lr=0.005, weight_decay=5e-4, 
                               patience=50, random_state=42):
    """
    训练GCN链接预测模型
    
    参数:
        data: PyG Data对象
        config_name: 'identity', 'topology', 'topology_weighted'
        use_edge_weight: 是否使用边权重
        test_size: 测试集比例
        val_size: 验证集比例
        epochs: 最大训练轮数
        lr: 学习率
        weight_decay: L2正则化系数
        patience: 早停耐心值
        random_state: 随机种子
    
    返回:
        results: dict 包含各指标和模型
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    
    # 根据配置设置节点特征
    if config_name == 'identity':
        x = torch.eye(num_nodes)
        print(f"配置: 单位矩阵 (无特征), 特征维度: {x.shape[1]}")
    elif config_name in ['topology', 'topology_weighted']:
        x = data.x
        print(f"配置: 拓扑特征 ({x.shape[1]}维)")
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    # 边权重
    edge_weight = data.edge_weight if use_edge_weight else None
    if use_edge_weight:
        print(f"使用边权重: combined_score")
    
    # 划分边集
    indices = list(range(num_edges))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    
    train_edges = data.edge_index[:, train_idx]
    test_edges = data.edge_index[:, test_idx]
    
    # 从训练集划分验证集
    num_train = train_edges.size(1)
    train_sub_idx, val_idx = train_test_split(list(range(num_train)), test_size=val_size, random_state=random_state)
    
    train_pos = train_edges[:, train_sub_idx]
    val_pos = train_edges[:, val_idx]
    test_pos = test_edges
    
    # 构建训练集边集合用于负采样
    train_edge_set = set()
    for i in range(train_edges.size(1)):
        u = train_edges[0, i].item()
        v = train_edges[1, i].item()
        train_edge_set.add((u, v))
        train_edge_set.add((v, u))
    
    # 采样负样本
    train_neg = sample_negative_edges(train_pos, num_nodes, train_pos.size(1), train_edge_set)
    val_neg = sample_negative_edges(val_pos, num_nodes, val_pos.size(1), train_edge_set)
    test_neg = sample_negative_edges(test_pos, num_nodes, test_pos.size(1), train_edge_set)
    
    # 合并正负样本
    train_edges_all = torch.cat([train_pos, train_neg], dim=1)
    train_labels = torch.cat([torch.ones(train_pos.size(1)), torch.zeros(train_neg.size(1))])
    
    val_edges_all = torch.cat([val_pos, val_neg], dim=1)
    val_labels = torch.cat([torch.ones(val_pos.size(1)), torch.zeros(val_neg.size(1))])
    
    test_edges_all = torch.cat([test_pos, test_neg], dim=1)
    test_labels = torch.cat([torch.ones(test_pos.size(1)), torch.zeros(test_neg.size(1))])
    
    print(f"训练集: {train_edges_all.size(1)} 条边 (正:{train_pos.size(1)}, 负:{train_neg.size(1)})")
    print(f"验证集: {val_edges_all.size(1)} 条边 (正:{val_pos.size(1)}, 负:{val_neg.size(1)})")
    print(f"测试集: {test_edges_all.size(1)} 条边 (正:{test_pos.size(1)}, 负:{test_neg.size(1)})")
    
    # 初始化模型
    in_dim = x.size(1)
    model = GCNLinkPredictor(in_dim=in_dim, hidden_dim=64, out_dim=32, dropout=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 训练
    best_val_auc = 0
    best_model = None
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(x, train_edges_all)
        pred = model.decode(z, train_edges_all)
        loss = F.binary_cross_entropy(pred, train_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, val_edges_all)
                pred = model.decode(z, val_edges_all)
                val_auc = roc_auc_score(val_labels.numpy(), pred.numpy())
                val_ap = average_precision_score(val_labels.numpy(), pred.numpy())
            
            print(f"Epoch {epoch:3d}: loss={loss.item():.4f}, val_auc={val_auc:.4f}, val_ap={val_ap:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 测试最佳模型
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        z = model.encode(x, test_edges_all)
        pred = model.decode(z, test_edges_all)
        test_auc = roc_auc_score(test_labels.numpy(), pred.numpy())
        test_ap = average_precision_score(test_labels.numpy(), pred.numpy())
    
    return {
        'config': config_name,
        'use_edge_weight': use_edge_weight,
        'test_auc': test_auc,
        'test_ap': test_ap,
        'best_val_auc': best_val_auc,
        'model_state': best_model
    }


def run_ablation_study(data):
    """运行消融实验"""
    print("=" * 60)
    print("消融实验：链接预测性能对比")
    print("=" * 60)
    
    configs = [
        ('identity', False, "单位矩阵（无特征）"),
        ('topology', False, "5个拓扑特征"),
        ('topology', True, "5个拓扑特征 + 边权重"),
    ]
    
    results = []
    for config_name, use_weight, desc in configs:
        print(f"\n>>> {desc}")
        print("-" * 40)
        result = train_gcn_link_prediction(
            data, 
            config_name=config_name,
            use_edge_weight=use_weight,
            epochs=300,
            patience=30
        )
        results.append(result)
        print(f"\n测试结果: AUC={result['test_auc']:.4f}, AP={result['test_ap']:.4f}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("消融实验汇总")
    print("=" * 60)
    print(f"{'配置':<30} {'AUC':<10} {'AP':<10}")
    print("-" * 50)
    for r in results:
        name = f"{r['config']}"
        if r['use_edge_weight']:
            name += " + edge_weight"
        print(f"{name:<30} {r['test_auc']:.4f}     {r['test_ap']:.4f}")
    
    return results


if __name__ == "__main__":
    data = torch.load("data/processed/graph.pt")
    print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
    print(f"特征维度: {data.x.shape[1]}")
    
    results = run_ablation_study(data)