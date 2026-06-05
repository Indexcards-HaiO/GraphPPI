# 模型

## 编码器

### GCNEncoder
两层 GCN 图卷积编码器，每层后接 BatchNorm + ReLU + Dropout。

参数: `in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.5`

### GATEncoder
多头图注意力编码器（heads=4），输出维度 `out_dim // heads`。

### SAGEEncoder
GraphSAGE 编码器，使用 mean 聚合。

### create_encoder(encoder_type, ...)
编码器工厂函数。支持 `'gcn'`, `'gat'`, `'sage'`。

## 解码器

### DotProductDecoder
直接计算节点嵌入的点积作为链接概率。

### MLPDecoder
将两个节点的嵌入拼接后通过 MLP 输出标量。

参数: `embed_dim, hidden_dim`

### EdgeFeatureMLPDecoder
MLP 解码器 + STRING 8 通道边特征。

参数: `embed_dim, edge_attr_dim=8, hidden_dim`

### create_decoder(decoder_type, ...)
解码器工厂函数。支持 `'dot'`, `'mlp'`, `'edge_mlp'`。

## 预测器

### LinkPredictor
组合编码器和解码器的完整链路预测模型。

参数: `in_dim, encoder_type, decoder_type, hidden_dim=128, out_dim=64, ...`
