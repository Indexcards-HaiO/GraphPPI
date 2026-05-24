#!/usr/bin/env python3
"""
解码器模块
- DotProductDecoder: 点积解码
- MLPDecoder: MLP 解码（只用节点嵌入）
- EdgeFeatureMLPDecoder: MLP 解码（拼接节点嵌入 + 边特征）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DotProductDecoder(nn.Module):
    """点积解码器 z_u · z_v → sigmoid"""

    def __init__(self):
        super().__init__()

    def forward(self, z, edge_index, edge_attr=None):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        return torch.sigmoid((z_u * z_v).sum(dim=-1))


class MLPDecoder(nn.Module):
    """MLP 解码器 [z_u || z_v] → MLP → sigmoid"""

    def __init__(self, embed_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, edge_index, edge_attr=None):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        edge_input = torch.cat([z_u, z_v], dim=-1)
        return self.mlp(edge_input).squeeze(-1)


class EdgeFeatureMLPDecoder(nn.Module):
    """边特征增强 MLP 解码器 [z_u || z_v || edge_attr] → MLP → sigmoid"""

    def __init__(self, embed_dim, edge_attr_dim=8, hidden_dim=64, dropout=0.5):
        super().__init__()
        input_dim = embed_dim * 2 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z, edge_index, edge_attr=None):
        z_u = z[edge_index[0]]
        z_v = z[edge_index[1]]
        if edge_attr is not None:
            edge_input = torch.cat([z_u, z_v, edge_attr], dim=-1)
        else:
            # 无 edge_attr 时用零填充
            edge_input = torch.cat([z_u, z_v,
                                     torch.zeros(z_u.size(0), 8, device=z.device)], dim=-1)
        return self.mlp(edge_input).squeeze(-1)


# 解码器工厂
DECODER_REGISTRY = {
    'dot': DotProductDecoder,
    'mlp': MLPDecoder,
    'edge_mlp': EdgeFeatureMLPDecoder,
}


def create_decoder(decoder_type, embed_dim, **kwargs):
    """创建解码器"""
    decoder_cls = DECODER_REGISTRY.get(decoder_type.lower(), MLPDecoder)
    if decoder_type.lower() == 'dot':
        return decoder_cls()
    return decoder_cls(embed_dim=embed_dim, **kwargs)
