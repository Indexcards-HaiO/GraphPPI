#!/usr/bin/env python3
"""
GNN 编码器模块
- GCNEncoder: 多层 GCN
- GATEncoder: 多层 GAT (Graph Attention)
- SAGEEncoder: 多层 GraphSAGE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCNEncoder(nn.Module):
    """多层 GCN 编码器"""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_dim, out_dim))
        else:
            self.convs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class GATEncoder(nn.Module):
    """多层 GAT 编码器"""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.5, heads=4):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout))
        else:
            self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim, out_dim, heads=1, dropout=dropout))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class SAGEEncoder(nn.Module):
    """多层 GraphSAGE 编码器"""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.convs.append(SAGEConv(hidden_dim, out_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


# 编码器工厂
ENCODER_REGISTRY = {
    'gcn': GCNEncoder,
    'gat': GATEncoder,
    'sage': SAGEEncoder,
}


def create_encoder(encoder_type, in_dim, **kwargs):
    """创建编码器"""
    encoder_cls = ENCODER_REGISTRY.get(encoder_type.lower(), GCNEncoder)
    return encoder_cls(in_dim=in_dim, **kwargs)
