# src/models/gcn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


class GCNLinkPredictor(nn.Module):
    """
    用于链接预测的GCN模型
    """
    
    def __init__(self, in_dim, hidden_dim=64, out_dim=32, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
    def encode(self, x, edge_index, edge_weight=None):
        """
        编码器：节点 → 嵌入
        """
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index):
        """
        解码器：计算边存在概率（点积）
        """
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return torch.sigmoid((z_u * z_v).sum(dim=-1))
    
    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        return z


class GCNLinkPredictorTrainer:
    """
    GCN链接预测模型的训练器
    """
    
    def __init__(self, model, lr=0.01, weight_decay=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def train_step(self, x, train_edges, train_labels, edge_weight=None):
        """
        单步训练
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        z = self.model.encode(x, train_edges, edge_weight)
        pred = self.model.decode(z, train_edges)
        
        loss = F.binary_cross_entropy(pred, train_labels)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, x, train_edges, train_labels, val_edges, val_labels,
              epochs=200, patience=20, edge_weight=None):
        """
        完整训练循环，包含早停
        """
        best_val_auc = 0
        best_epoch = 0
        wait = 0
        
        for epoch in range(epochs):
            loss = self.train_step(x, train_edges, train_labels, edge_weight)
            
            if epoch % 20 == 0:
                val_auc, val_ap = self.evaluate(x, val_edges, val_labels, edge_weight)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_epoch = epoch
                    wait = 0
                    # 保存最佳模型
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    wait += 1
                
                if epoch % 50 == 0:
                    print(f"Epoch {epoch}: loss={loss:.4f}, val_auc={val_auc:.4f}, val_ap={val_ap:.4f}")
                
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}, best val_auc={best_val_auc:.4f} at epoch {best_epoch}")
                    break
        
        # 恢复最佳模型
        self.model.load_state_dict(best_state)
        return best_val_auc
    
    def evaluate(self, x, test_edges, test_labels, edge_weight=None):
        """
        评估模型
        """
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(x, test_edges, edge_weight)
            pred = self.model.decode(z, test_edges)
            
            pred_np = pred.cpu().numpy()
            labels_np = test_labels.cpu().numpy()
            
            auc = roc_auc_score(labels_np, pred_np)
            ap = average_precision_score(labels_np, pred_np)
            
        return auc, ap