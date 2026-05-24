"""测试训练器"""
import torch
import numpy as np
from graphppi.models.predictor import LinkPredictor
from graphppi.trainer import LinkPredictionTrainer


def test_trainer_creation():
    """训练器创建"""
    model = LinkPredictor(in_dim=5, encoder_type='gcn', decoder_type='dot',
                          hidden_dim=16, out_dim=8)
    trainer = LinkPredictionTrainer(model, lr=0.01, weight_decay=1e-4)
    assert trainer.optimizer is not None


def test_train_step():
    """单步训练"""
    np.random.seed(42)
    torch.manual_seed(42)

    x = torch.randn(5, 3)
    mp_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    train_edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    train_labels = torch.tensor([1.0, 0.0])

    model = LinkPredictor(in_dim=3, encoder_type='gcn', decoder_type='dot',
                          hidden_dim=16, out_dim=8, num_layers=2)
    trainer = LinkPredictionTrainer(model, lr=0.01)

    loss = trainer.train_step(x, mp_edge_index, train_edges, train_labels)
    assert isinstance(loss, float)
    assert loss > 0


def test_evaluate():
    """评估"""
    np.random.seed(42)
    torch.manual_seed(42)

    x = torch.randn(5, 3)
    mp_edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    eval_edges = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    eval_labels = torch.tensor([1.0, 0.0, 1.0])

    model = LinkPredictor(in_dim=3, encoder_type='gcn', decoder_type='dot',
                          hidden_dim=16, out_dim=8, num_layers=2)
    trainer = LinkPredictionTrainer(model, lr=0.01)
    metrics = trainer.evaluate(x, mp_edge_index, eval_edges, eval_labels)
    assert 'auc' in metrics
    assert 'ap' in metrics
