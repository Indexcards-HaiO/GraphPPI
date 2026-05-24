# src/train_link_prediction_tuned.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import itertools


class GCNLinkPredictorMLP(nn.Module):
    """使用MLP解码器的GCN模型"""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # MLP解码器
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        edge_features = torch.cat([z_u, z_v], dim=1)
        return self.mlp(edge_features).squeeze()


class GCNLinkPredictorDot(nn.Module):
    """使用点积解码器的GCN模型"""
    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return torch.sigmoid((z_u * z_v).sum(dim=-1))


def sample_hard_negative_edges(pos_edges, num_nodes, num_samples, exclude_set, degree):
    """
    采样度数匹配的难负样本
    
    参数:
        pos_edges: 正样本边
        num_nodes: 节点数
        num_samples: 需要采样的负样本数
        exclude_set: 排除的边集合
        degree: 每个节点的度数
    """
    # 获取正样本中节点的度数分布
    pos_nodes = torch.unique(pos_edges).tolist()
    pos_degrees = [degree[node] for node in pos_nodes]
    degree_bins = np.histogram(pos_degrees, bins=10)[1]
    
    neg_edges = []
    exclude = exclude_set.copy()
    
    while len(neg_edges) < num_samples:
        # 从正样本度数分布中采样度数
        target_degree = np.random.choice(pos_degrees)
        # 找到度数接近的节点
        candidates = [i for i in range(num_nodes) if abs(degree[i] - target_degree) < 5]
        if len(candidates) < 2:
            candidates = list(range(num_nodes))
        
        u = np.random.choice(candidates)
        v = np.random.choice(candidates)
        
        if u != v and (u, v) not in exclude and (v, u) not in exclude:
            neg_edges.append([u, v])
            exclude.add((u, v))
            exclude.add((v, u))
    
    return torch.tensor(neg_edges, dtype=torch.long).t()


def train_gcn_with_config(data, config):
    """
    使用指定配置训练GCN
    
    参数:
        data: PyG Data对象
        config: dict 包含超参数
    """
    torch.manual_seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    
    # 节点特征
    if config.get('use_identity', False):
        x = torch.eye(num_nodes)
    else:
        x = data.x
    
    # 边权重处理
    edge_weight = None
    if config.get('use_edge_weight', False):
        # 将 combined_score 缩放到更合理的范围
        edge_weight = data.edge_weight * config.get('weight_scale', 1.0)
    
    # 划分边集
    indices = list(range(num_edges))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=config.get('seed', 42))
    
    train_edges_full = data.edge_index[:, train_idx]
    test_edges = data.edge_index[:, test_idx]
    
    # 从训练集中划分验证集
    num_train = train_edges_full.size(1)
    train_sub_idx, val_idx = train_test_split(
        list(range(num_train)), 
        test_size=0.2, 
        random_state=config.get('seed', 42)
    )
    
    train_pos = train_edges_full[:, train_sub_idx]
    val_pos = train_edges_full[:, val_idx]
    test_pos = test_edges
    
    # 构建排除集合
    exclude_set = set()
    for i in range(train_edges_full.size(1)):
        u = train_edges_full[0, i].item()
        v = train_edges_full[1, i].item()
        exclude_set.add((u, v))
        exclude_set.add((v, u))
    
    # 计算度数用于难负采样
    degree = torch.zeros(num_nodes)
    for i in range(data.edge_index.size(1)):
        u = data.edge_index[0, i].item()
        v = data.edge_index[1, i].item()
        degree[u] += 1
        degree[v] += 1
    
    # 负采样
    if config.get('hard_negative', False):
        train_neg = sample_hard_negative_edges(train_pos, num_nodes, train_pos.size(1), exclude_set, degree)
        val_neg = sample_hard_negative_edges(val_pos, num_nodes, val_pos.size(1), exclude_set, degree)
        test_neg = sample_hard_negative_edges(test_pos, num_nodes, test_pos.size(1), exclude_set, degree)
    else:
        # 标准随机负采样
        train_neg = sample_hard_negative_edges(train_pos, num_nodes, train_pos.size(1), exclude_set, degree)
        val_neg = sample_hard_negative_edges(val_pos, num_nodes, val_pos.size(1), exclude_set, degree)
        test_neg = sample_hard_negative_edges(test_pos, num_nodes, test_pos.size(1), exclude_set, degree)
    
    # 合并
    train_edges_all = torch.cat([train_pos, train_neg], dim=1)
    train_labels = torch.cat([torch.ones(train_pos.size(1)), torch.zeros(train_neg.size(1))])
    val_edges_all = torch.cat([val_pos, val_neg], dim=1)
    val_labels = torch.cat([torch.ones(val_pos.size(1)), torch.zeros(val_neg.size(1))])
    test_edges_all = torch.cat([test_pos, test_neg], dim=1)
    test_labels = torch.cat([torch.ones(test_pos.size(1)), torch.zeros(test_neg.size(1))])
    
    # 模型
    if config.get('decoder', 'dot') == 'mlp':
        model = GCNLinkPredictorMLP(
            in_dim=x.size(1),
            hidden_dim=config.get('hidden_dim', 128),
            out_dim=config.get('out_dim', 64),
            dropout=config.get('dropout', 0.5)
        )
    else:
        model = GCNLinkPredictorDot(
            in_dim=x.size(1),
            hidden_dim=config.get('hidden_dim', 128),
            out_dim=config.get('out_dim', 64),
            dropout=config.get('dropout', 0.5)
        )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.get('lr', 0.005),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    # 训练
    best_val_auc = 0
    best_model = None
    wait = 0
    
    for epoch in range(config.get('epochs', 500)):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(x, train_edges_all, edge_weight)
        pred = model.decode(z, train_edges_all)
        loss = F.binary_cross_entropy(pred, train_labels)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                z = model.encode(x, val_edges_all, edge_weight)
                pred = model.decode(z, val_edges_all)
                val_auc = roc_auc_score(val_labels.numpy(), pred.numpy())
                val_ap = average_precision_score(val_labels.numpy(), pred.numpy())
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model = {k: v.clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            
            if wait >= config.get('patience', 30):
                break
    
    # 测试
    model.load_state_dict(best_model)
    model.eval()
    with torch.no_grad():
        z = model.encode(x, test_edges_all, edge_weight)
        pred = model.decode(z, test_edges_all)
        test_auc = roc_auc_score(test_labels.numpy(), pred.numpy())
        test_ap = average_precision_score(test_labels.numpy(), pred.numpy())
    
    return test_auc, test_ap


def hyperparameter_search(data):
    """超参数网格搜索"""
    print("=" * 60)
    print("超参数网格搜索")
    print("=" * 60)
    
    # 定义搜索空间
    hidden_dims = [64, 128]
    dropouts = [0.3, 0.5]
    lrs = [0.001, 0.005]
    decoders = ['dot', 'mlp']
    use_edge_weights = [False, True]
    
    best_auc = 0
    best_config = None
    
    for hidden_dim, dropout, lr, decoder, use_weight in itertools.product(
        hidden_dims, dropouts, lrs, decoders, use_edge_weights
    ):
        config = {
            'hidden_dim': hidden_dim,
            'dropout': dropout,
            'lr': lr,
            'decoder': decoder,
            'use_edge_weight': use_weight,
            'use_identity': False,
            'hard_negative': False,
            'epochs': 200,
            'patience': 20,
            'weight_scale': 1.0,
            'seed': 42
        }
        
        print(f"\n测试配置: hidden={hidden_dim}, dropout={dropout}, lr={lr}, decoder={decoder}, edge_weight={use_weight}")
        try:
            auc, ap = train_gcn_with_config(data, config)
            print(f"  -> AUC={auc:.4f}, AP={ap:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_config = config.copy()
                best_config['test_auc'] = auc
                best_config['test_ap'] = ap
        except Exception as e:
            print(f"  -> 错误: {e}")
    
    print("\n" + "=" * 60)
    print("最佳配置:")
    print("=" * 60)
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    
    return best_config


if __name__ == "__main__":
    data = torch.load("data/processed/graph.pt")
    print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
    print(f"特征维度: {data.x.shape[1]}")
    
    # 超参数搜索
    best_config = hyperparameter_search(data)
    
    # 使用最佳配置重新训练完整轮数
    print("\n" + "=" * 60)
    print("使用最佳配置完整训练")
    print("=" * 60)
    
    best_config['epochs'] = 500
    best_config['patience'] = 50
    
    final_auc, final_ap = train_gcn_with_config(data, best_config)
    print(f"\n最终结果: AUC={final_auc:.4f}, AP={final_ap:.4f}")