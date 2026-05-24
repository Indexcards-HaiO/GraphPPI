"""测试 Encoder / Decoder / Predictor 创建和前向传播"""
import torch
from graphppi.models.encoder import GCNEncoder, GATEncoder, SAGEEncoder, create_encoder
from graphppi.models.decoder import DotProductDecoder, MLPDecoder, EdgeFeatureMLPDecoder, create_decoder
from graphppi.models.predictor import LinkPredictor


def test_gcn_encoder():
    """GCN 编码器前向传播"""
    x = torch.randn(5, 3)  # 5节点, 3维特征
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    encoder = GCNEncoder(in_dim=3, hidden_dim=8, out_dim=4, num_layers=2)
    z = encoder(x, edge_index)
    assert z.shape == (5, 4)


def test_gat_encoder():
    """GAT 编码器前向传播"""
    x = torch.eye(5)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    encoder = GATEncoder(in_dim=5, hidden_dim=8, out_dim=4, num_layers=1)
    z = encoder(x, edge_index)
    assert z.shape == (5, 4)


def test_sage_encoder():
    """SAGE 编码器前向传播"""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    encoder = SAGEEncoder(in_dim=3, hidden_dim=8, out_dim=4, num_layers=1)
    z = encoder(x, edge_index)
    assert z.shape == (5, 4)


def test_decoders():
    """三种解码器前向传播"""
    z = torch.randn(4, 64)  # 4节点, 64维嵌入
    edge_index = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)

    # Dot
    dot = DotProductDecoder()
    p_dot = dot(z, edge_index)
    assert p_dot.shape == (2,)
    assert (0 <= p_dot).all() and (p_dot <= 1).all()

    # MLP
    mlp = MLPDecoder(embed_dim=64, hidden_dim=32)
    p_mlp = mlp(z, edge_index)
    assert p_mlp.shape == (2,)

    # EdgeMLP
    emlp = EdgeFeatureMLPDecoder(embed_dim=64, edge_attr_dim=8, hidden_dim=32)
    edge_attr = torch.randn(2, 8)
    p_emlp = emlp(z, edge_index, edge_attr)
    assert p_emlp.shape == (2,)


def test_create_encoder_decoder():
    """工厂函数测试"""
    enc = create_encoder('gcn', in_dim=5, hidden_dim=8, out_dim=4)
    assert isinstance(enc, GCNEncoder)

    dec = create_decoder('dot', embed_dim=4)
    assert isinstance(dec, DotProductDecoder)


def test_link_predictor():
    """LinkPredictor 完整前向传播"""
    x = torch.randn(3, 5)
    mp_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pred_edge_index = torch.tensor([[0, 2]], dtype=torch.long)

    model = LinkPredictor(in_dim=5, encoder_type='gcn', decoder_type='mlp',
                          hidden_dim=16, out_dim=8, num_layers=2)
    pred = model(x, mp_edge_index, pred_edge_index)
    assert pred.shape == (1,)
    assert 0 <= pred.item() <= 1
