import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, ReLU, Linear
from torch_geometric.nn import MLP, GINConv, DeepGCNLayer
from sklearn.metrics import f1_score


class SMNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_vec_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(1, num_layers + 1):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            conv = GINConv(nn=mlp, train_eps=True)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.2, ckpt_grad=k % 3)
            self.layers.append(layer)
            in_channels = hidden_channels

    def forward(self, feature_vector: Tensor, adj_index: Tensor, edge_vector: Tensor):
        x = self.layers[0].conv(feature_vector, adj_index)

        for layer in self.layers[1:]:
            x = layer(x, adj_index)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.2, training=self.training)

        return x


class Classifier(nn.Module):
    def __init__(self, hidden_channels, class_num):
        super(Classifier, self).__init__()
        self.fc = Linear(hidden_channels, class_num)

    def forward(self, x):
        x = self.fc(x)

        return x


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1_result = f1_score(labels, preds, average='macro')

    return f1_result
