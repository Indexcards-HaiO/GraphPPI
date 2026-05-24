# src/baselines/node2vec_rf.py

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx


class Node2VecRF:
    """
    Node2Vec + 随机森林的链接预测模型
    """
    
    def __init__(self, dimensions=64, walk_length=30, num_walks=200, 
                 p=1.0, q=1.0, window=10, n_estimators=100, random_state=42):
        """
        参数:
            dimensions: 嵌入维度
            walk_length: 随机游走长度
            num_walks: 每个节点随机游走次数
            p: 返回参数
            q: 进出参数
            window: skip-gram窗口大小
            n_estimators: 随机森林树的数量
            random_state: 随机种子
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.node2vec = None
        self.rf = None
        self.embeddings = None
    
    def fit(self, train_edges, num_nodes, train_edge_pairs, train_labels):
        """
        训练Node2Vec嵌入和随机森林分类器
        
        参数:
            train_edges: tensor (2, num_edges) 训练集边（用于Node2Vec）
            num_nodes: int
            train_edge_pairs: tensor (2, num_train_pairs) 用于训练的边对
            train_labels: tensor (num_train_pairs,) 训练标签
        """
        # 构建networkx图
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        # 添加边（无向）
        for i in range(train_edges.size(1)):
            u = train_edges[0, i].item()
            v = train_edges[1, i].item()
            G.add_edge(u, v)
        
        # 使用Node2Vec（通过node2vec库）
        try:
            from node2vec import Node2Vec
            self.node2vec = Node2Vec(G, dimensions=self.dimensions, walk_length=self.walk_length,
                                      num_walks=self.num_walks, p=self.p, q=self.q,
                                      workers=1, seed=self.random_state)
            model = self.node2vec.fit(window=self.window, seed=self.random_state)
            
            # 获取节点嵌入
            self.embeddings = np.zeros((num_nodes, self.dimensions))
            for node in range(num_nodes):
                self.embeddings[node] = model.wv[str(node)]
        except ImportError:
            # 如果node2vec未安装，使用随机嵌入作为后备
            print("警告: node2vec未安装，使用随机嵌入")
            np.random.seed(self.random_state)
            self.embeddings = np.random.randn(num_nodes, self.dimensions)
        
        # 构建边特征（拼接两个节点的嵌入）
        X_train = []
        for i in range(train_edge_pairs.size(1)):
            u = train_edge_pairs[0, i].item()
            v = train_edge_pairs[1, i].item()
            edge_feat = np.concatenate([self.embeddings[u], self.embeddings[v]])
            X_train.append(edge_feat)
        
        X_train = np.array(X_train)
        y_train = train_labels.numpy()
        
        # 训练随机森林
        self.rf = RandomForestClassifier(n_estimators=self.n_estimators, 
                                          random_state=self.random_state,
                                          n_jobs=1)
        self.rf.fit(X_train, y_train)
    
    def predict(self, test_edges):
        """
        预测测试边的存在概率
        
        参数:
            test_edges: tensor (2, num_test_edges)
        
        返回:
            scores: array 预测概率
        """
        X_test = []
        for i in range(test_edges.size(1)):
            u = test_edges[0, i].item()
            v = test_edges[1, i].item()
            edge_feat = np.concatenate([self.embeddings[u], self.embeddings[v]])
            X_test.append(edge_feat)
        
        X_test = np.array(X_test)
        scores = self.rf.predict_proba(X_test)[:, 1]
        
        return scores
    
    def evaluate(self, test_edges, test_labels):
        """
        评估模型
        
        返回:
            auc: float
            ap: float
            scores: array
        """
        scores = self.predict(test_edges)
        auc = roc_auc_score(test_labels.numpy(), scores)
        ap = average_precision_score(test_labels.numpy(), scores)
        
        return auc, ap, scores