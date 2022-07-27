import torch
import torch.nn as nn
from torch_geometric.nn.inits import uniform

from .auto_encoder import AutoEncoder, VariationalAutoEncoder
from .molecule_gnn_model import GNN, GNN_graphpred
from .schnet import SchNet
from .dti_model import ProteinModel, MoleculeProteinModel


class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim=1)
