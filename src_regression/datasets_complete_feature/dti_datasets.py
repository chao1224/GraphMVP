import os

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset

from .molecule_datasets import mol_to_graph_data_obj_simple

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000


def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x


class MoleculeProteinDataset(InMemoryDataset):
    def __init__(self, root, dataset, mode):
        super(InMemoryDataset, self).__init__()
        self.root = root
        self.dataset = dataset
        datapath = os.path.join(self.root, self.dataset, '{}.csv'.format(mode))
        print('datapath\t', datapath)

        self.process_molecule()
        self.process_protein()

        df = pd.read_csv(datapath)
        self.molecule_index_list = df['smiles_id'].tolist()
        self.protein_index_list = df['target_id'].tolist()
        self.label_list = df['affinity'].tolist()
        self.label_list = torch.FloatTensor(self.label_list)

        return

    def process_molecule(self):
        input_path = os.path.join(self.root, self.dataset, 'smiles.csv')
        input_df = pd.read_csv(input_path, sep=',')
        smiles_list = input_df['smiles']

        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in rdkit_mol_objs_list]
        preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in preprocessed_rdkit_mol_objs_list]
        assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
        assert len(smiles_list) == len(preprocessed_smiles_list)

        smiles_list, rdkit_mol_objs = preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list

        data_list = []
        for i in range(len(smiles_list)):
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol != None:
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data_list.append(data)

        self.molecule_list = data_list
        return

    def process_protein(self):
        datapath = os.path.join(self.root, self.dataset, 'protein.csv')

        input_df = pd.read_csv(datapath, sep=',')
        protein_list = input_df['protein'].tolist()

        self.protein_list = [seq_cat(t) for t in protein_list]
        self.protein_list = torch.LongTensor(self.protein_list)
        return

    def __getitem__(self, idx):
        molecule = self.molecule_list[self.molecule_index_list[idx]]
        protein = self.protein_list[self.protein_index_list[idx]]
        label = self.label_list[idx]
        return molecule, protein, label

    def __len__(self):
        return len(self.label_list)
