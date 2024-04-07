import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import LayerNorm, Linear
from torch_geometric.nn import MLP, GINEConv
from sklearn.metrics import f1_score


class SMNetEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, edge_vec_dim):
        super().__init__()

        self.layers = nn.ModuleList()
        for k in range(1, num_layers + 1):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            conv = GINEConv(nn=mlp, train_eps=True, edge_dim=edge_vec_dim)
            self.layers.append(conv)
            in_channels = hidden_channels
        
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

    def forward(self, feature_vector: Tensor, adj_index: Tensor, edge_vector: Tensor):
        x = self.layers[0](feature_vector, adj_index, edge_vector)

        for layer in self.layers[1:]:
            x = F.relu(self.norm(x))
            x = F.dropout(x, p=0.2, training=self.training)
            x = layer(x, adj_index, edge_vector)
        
        x = F.relu(self.norm(x))
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
