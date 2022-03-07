import torch
import torch.nn as nn
import torch.nn.functional as F
from .schnet import SchNet
from .infograph import Discriminator
from .molecule_gnn_model import GNN, GNN_graphpred
from .auto_encoder import AutoEncoder, VariationalAutoEncoder


# class AliasTable(nn.Module):
#     def __init__(self, probs):
#         super(AliasTable, self).__init__()
#         self.num_element = len(probs)
#         probs, alias = self.build(probs)
#         self.register_buffer("probs", probs)
#         self.register_buffer("alias", alias)
#         self.device = 'cpu'

#     def build(self, probs):
#         with torch.no_grad():
#             probs = probs / probs.mean()
#             alias = torch.zeros_like(probs, dtype=torch.long)

#             index = torch.arange(len(probs))
#             is_available = probs < 1
#             small_index = index[is_available]
#             large_index = index[~is_available]
#             while len(small_index) and len(large_index):
#                 count = min(len(small_index), len(large_index))
#                 small, small_index = small_index.split((count, len(small_index) - count))
#                 large, large_index = large_index.split((count, len(large_index) - count))

#                 alias[small] = large
#                 probs[large] += probs[small] - 1

#                 is_available = probs[large] < 1
#                 small_index_new = large[is_available]
#                 large_index_new = large[~is_available]
#                 small_index = torch.cat([small_index, small_index_new])
#                 large_index = torch.cat([large_index, large_index_new])

#             alias[small_index] = small_index
#             alias[large_index] = large_index

#         return probs, alias

#     def sample(self, sample_shape):
#         with torch.no_grad():
#             index = torch.randint(self.num_element, sample_shape, device=self.device)
#             prob = torch.rand(sample_shape, device=self.device)
#             samples = torch.where(prob < self.probs[index], index, self.alias[index])

#         return samples


# class GCNNet(torch.nn.Module):
#     def __init__(self, num_node, embedding_dim=300, hidden_dim=300):
#         super(GCNNet, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
#         self.conv1 = GCNConv(embedding_dim, hidden_dim, cached=True, normalize=True)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim, cached=True, normalize=True)

#     def forward(self, data):
#         x = self.embedding(data.x)
#         x = self.conv1(x, data.edge_index)
#         x = self.conv2(x, data.edge_index)
#         return x


# class GATNet(torch.nn.Module):
#     def __init__(self, num_node, embedding_dim=300, hidden_dim=300, num_head=8):
#         super(GATNet, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
#         self.conv1 = GATConv(embedding_dim, hidden_dim, heads=num_head, dropout=0.6)
#         self.conv2 = GATConv(hidden_dim * num_head, hidden_dim, heads=1, concat=False, dropout=0.6)

#     def forward(self, data):
#         x = self.embedding(data.x)
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, data.edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, data.edge_index)
#         return x


# class Matching_Classifier(nn.Sequential):
#     def __init__(self, input_dim_drug=300, input_dim_protein=300):
#         super(Matching_Classifier, self).__init__()

#         self.input_dim_drug = input_dim_drug
#         self.input_dim_protein = input_dim_protein
#         self.dropout = nn.Dropout(0.1)
#         self.hidden_dims = [1024, 512]
#         self.layer_size = len(self.hidden_dims) + 1
#         dims = [self.input_dim_drug + self.input_dim_protein] + \
#                self.hidden_dims + [1]

#         self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
#                                         for i in range(self.layer_size)])
#         self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1])
#                                           for i in range(self.layer_size)])
#         return

#     def forward(self, emb_0, emb_1):
#         v = torch.cat([emb_0, emb_1], 1)
#         for i, l in enumerate(self.predictor):
#             if i == self.layer_size - 1:
#                 v = l(v)
#             else:
#                 v = l(v)
#                 v = self.batch_norms[i](v)
#                 v = F.relu(v)
#                 v = self.dropout(v)

#         v = v.squeeze()
#         return v
