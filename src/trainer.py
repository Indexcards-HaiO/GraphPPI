#!/usr/bin/env python3
"""
链路预测训练器
- 训练循环（消息传递严格隔离）
- 早停机制
- 模型保存/加载
"""

import torch
import torch.nn.functional as F
import copy

try:
    from graphppi.metrics import compute_all_metrics
except ModuleNotFoundError:
    from metrics import compute_all_metrics


class LinkPredictionTrainer:
    """链路预测训练器"""

    def __init__(self, model, lr=0.005, weight_decay=1e-4, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.best_state = None
        self.best_val_auc = 0
        self.best_epoch = 0

    def train_step(self, x, mp_edge_index, train_edges, train_labels,
                   mp_edge_weight=None, train_edge_attr=None):
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        # 消息传递只用训练正边
        z = self.model.encode(
            x.to(self.device),
            mp_edge_index.to(self.device),
            mp_edge_weight.to(self.device) if mp_edge_weight is not None else None
        )

        # 预测
        pred = self.model.decode(
            z,
            train_edges.to(self.device),
            train_edge_attr.to(self.device) if train_edge_attr is not None else None
        )

        loss = F.binary_cross_entropy(pred, train_labels.to(self.device))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, x, mp_edge_index, eval_edges, eval_labels,
                 mp_edge_weight=None, eval_edge_attr=None):
        """评估（消息传递只用训练边，预测用评估边）"""
        self.model.eval()
        with torch.no_grad():
            z = self.model.encode(
                x.to(self.device),
                mp_edge_index.to(self.device),
                mp_edge_weight.to(self.device) if mp_edge_weight is not None else None
            )
            pred = self.model.decode(
                z,
                eval_edges.to(self.device),
                eval_edge_attr.to(self.device) if eval_edge_attr is not None else None
            )
            metrics = compute_all_metrics(pred.cpu(), eval_labels.cpu())
        return metrics

    def train(self, x, mp_edge_index,
              train_edges, train_labels,
              val_edges, val_labels,
              epochs=500, patience=50,
              mp_edge_weight=None,
              train_edge_attr=None,
              val_edge_attr=None,
              verbose=True, log_interval=50):
        """
        完整训练循环
        
        关键设计：消息传递图 mp_edge_index 在整个训练/验证过程中
        始终只包含训练正边，验证边不参与消息传递。
        """
        best_val_auc = 0
        best_state = None
        wait = 0

        for epoch in range(epochs):
            loss = self.train_step(
                x, mp_edge_index, train_edges, train_labels,
                mp_edge_weight, train_edge_attr
            )

            if epoch % log_interval == 0:
                val_metrics = self.evaluate(
                    x, mp_edge_index, val_edges, val_labels,
                    mp_edge_weight, val_edge_attr
                )
                val_auc = val_metrics['auc']

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = copy.deepcopy(self.model.state_dict())
                    wait = 0
                    self.best_epoch = epoch
                else:
                    wait += 1

                if verbose:
                    print(f"  Epoch {epoch:4d}: loss={loss:.4f}, val_auc={val_auc:.4f}, "
                          f"val_ap={val_metrics['ap']:.4f}, "
                          f"val_hits@10={val_metrics.get('hits@10', 0):.4f}")

                if wait >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch}, best val_auc={best_val_auc:.4f}")
                    break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.best_val_auc = best_val_auc
        else:
            self.best_val_auc = 0

        return best_val_auc

    def test(self, x, mp_edge_index, test_edges, test_labels,
             mp_edge_weight=None, test_edge_attr=None):
        """测试（消息传递只用训练边）"""
        return self.evaluate(
            x, mp_edge_index, test_edges, test_labels,
            mp_edge_weight, test_edge_attr
        )

    def get_model(self):
        """获取当前模型（CPU）"""
        return self.model.cpu()
