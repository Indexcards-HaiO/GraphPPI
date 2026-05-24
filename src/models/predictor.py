#!/usr/bin/env python3
"""
统一链路预测模型
组合 encoder + decoder
"""

import torch
import torch.nn as nn
from .encoder import create_encoder
from .decoder import create_decoder


class LinkPredictor(nn.Module):
    """链路预测模型 = Encoder + Decoder"""

    def __init__(self, in_dim, encoder_type='gcn', decoder_type='dot',
                 hidden_dim=128, out_dim=64, num_layers=2, dropout=0.5, **kwargs):
        super().__init__()

        self.encoder = create_encoder(
            encoder_type, in_dim,
            hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, dropout=dropout, **kwargs
        )

        self.decoder = create_decoder(
            decoder_type, out_dim,
            hidden_dim=hidden_dim // 2, dropout=dropout, **kwargs
        )

    def encode(self, x, edge_index, edge_weight=None):
        """编码：节点特征 → 节点嵌入"""
        return self.encoder(x, edge_index, edge_weight)

    def decode(self, z, edge_index, edge_attr=None):
        """解码：节点嵌入 + 边 → 链接概率"""
        return self.decoder(z, edge_index, edge_attr)

    def forward(self, x, mp_edge_index, pred_edge_index,
                mp_edge_weight=None, pred_edge_attr=None):
        """
        前向传播
        
        参数:
            x: 节点特征
            mp_edge_index: 消息传递用的边（只应包含训练正边）
            pred_edge_index: 要预测的边对
            mp_edge_weight: 消息传递用的边权重
            pred_edge_attr: 预测用的边特征
        """
        z = self.encode(x, mp_edge_index, mp_edge_weight)
        return self.decode(z, pred_edge_index, pred_edge_attr)
