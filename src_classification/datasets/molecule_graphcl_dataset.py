
import os
from itertools import repeat

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from .molecule_datasets import MoleculeDataset


class MoleculeDataset_graphcl(MoleculeDataset):

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 empty=False):

        self.aug_prob = None
        self.aug_mode = 'no_aug'
        self.aug_strength = 0.2
        self.augmentations = [self.node_drop, self.subgraph,
                              self.edge_pert, self.attr_mask, lambda x: x]
        super(MoleculeDataset_graphcl, self).__init__(
            root, transform, pre_transform, pre_filter, dataset, empty)

    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode

    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def set_augProb(self, aug_prob):
        self.aug_prob = aug_prob

    def node_drop(self, data):

        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_strength)

        idx_perm = np.random.permutation(node_num)
        idx_nodrop = idx_perm[drop_num:].tolist()
        idx_nodrop.sort()

        edge_idx, edge_attr = subgraph(subset=idx_nodrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nodrop]
        data.__num_nodes__, _ = data.x.shape
        return data

    def edge_pert(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num),
                                    replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_attr = data.edge_attr[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        # edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        edge_index_nonexist = torch.nonzero(adj, as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1],
                                   pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        # random 4-class & 3-class edge_attr for 1st & 2nd dimension
        edge_attr_add_1 = torch.tensor(np.random.randint(
            4, size=(edge_index_add.shape[1], 1)))
        edge_attr_add_2 = torch.tensor(np.random.randint(
            3, size=(edge_index_add.shape[1], 1)))
        edge_attr_add = torch.cat((edge_attr_add_1, edge_attr_add_2), dim=1)
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    def attr_mask(self, data):

        _x = data.x.clone()
        node_num, _ = data.x.size()
        mask_num = int(node_num * self.aug_strength)

        token = data.x.float().mean(dim=0).long()
        idx_mask = np.random.choice(
            node_num, mask_num, replace=False)

        _x[idx_mask] = token
        data.x = _x
        return data

    def subgraph(self, data):

        G = to_networkx(data)
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1 - self.aug_strength))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()

        edge_idx, edge_attr = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape
        return data

    def get(self, idx):
        data, data1, data2 = Data(), Data(), Data()
        keys_for_2D = ['x', 'edge_index', 'edge_attr']
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            if key in keys_for_2D:
                data[key], data1[key], data2[key] = item[s], item[s], item[s]
            else:
                data[key] = item[s]

        if self.aug_mode == 'no_aug':
            n_aug1, n_aug2 = 4, 4
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'sample':
            n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        else:
            raise ValueError
        return data, data1, data2
