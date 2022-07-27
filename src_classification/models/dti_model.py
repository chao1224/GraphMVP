import torch
from torch import nn
from torch_geometric.nn import global_mean_pool


class ProteinModel(nn.Module):
    def __init__(self, emb_dim=128, num_features=25, output_dim=128, n_filters=32, kernel_size=8):
        super(ProteinModel, self).__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.intermediate_dim = emb_dim - kernel_size + 1

        self.embedding = nn.Embedding(num_features+1, emb_dim)
        self.n_filters = n_filters
        self.conv1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=kernel_size)
        self.fc = nn.Linear(n_filters*self.intermediate_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.conv1(x)
        x = x.view(-1, self.n_filters*self.intermediate_dim)
        x = self.fc(x)
        return x


class MoleculeProteinModel(nn.Module):
    def __init__(self, molecule_model, protein_model, molecule_emb_dim, protein_emb_dim, output_dim=1, dropout=0.2):
        super(MoleculeProteinModel, self).__init__()
        self.fc1 = nn.Linear(molecule_emb_dim+protein_emb_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, output_dim)
        self.molecule_model = molecule_model
        self.protein_model = protein_model
        self.pool = global_mean_pool
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, molecule, protein):
        molecule_node_representation = self.molecule_model(molecule)
        molecule_representation = self.pool(molecule_node_representation, molecule.batch)
        protein_representation = self.protein_model(protein)

        x = torch.cat([molecule_representation, protein_representation], dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)

        return x
