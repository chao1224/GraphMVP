import random

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph
from tqdm import tqdm


def search_graph(graph):
    num_node = len(graph.x)
    edge_set = set()

    u_list, v_list = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
    for u,v in zip(u_list, v_list):
        edge_set.add((u,v))
        edge_set.add((v,u))
    
    visited_list = []
    unvisited_set = set([i for i in range(num_node)])

    while len(unvisited_set) > 0:
        u = random.sample(unvisited_set, 1)[0]
        queue = [u]
        while len(queue):
            u = queue.pop(0)
            if u in visited_list:
                continue
            visited_list.append(u)
            unvisited_set.remove(u)

            for v in range(num_node):
                if (v not in visited_list) and ((u,v) in edge_set):
                    queue.append(v)
    assert len(visited_list) == num_node
    return visited_list


class MoleculeDatasetGPT(InMemoryDataset):
    def __init__(self, molecule_dataset, transform=None, pre_transform=None):
        self.molecule_dataset = molecule_dataset
        self.root = molecule_dataset.root + '_GPT'
        super(MoleculeDatasetGPT, self).__init__(self.root, transform=transform, pre_transform=pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        return

    def process(self):
        num_molecule = len(self.molecule_dataset)
        data_list = []
        for i in tqdm(range(num_molecule)):
            graph = self.molecule_dataset.get(i)

            num_node = len(graph.x)
            # TODO: will replace this with DFS/BFS searching
            node_list = search_graph(graph)

            for idx in range(num_node-1):
                # print('sub_node_list: {}\nnext_node: {}'.format(sub_node_list, next_node))
                # [0..idx] -> [idx+1]
                sub_node_list = node_list[:idx+1]
                next_node = node_list[idx+1]

                edge_index, edge_attr = subgraph(
                    subset=sub_node_list, edge_index=graph.edge_index, edge_attr=graph.edge_attr,
                    relabel_nodes=True, num_nodes=num_node)

                # Take the subgraph and predict on the next node (atom type only)
                sub_graph = Data(x=graph.x[sub_node_list], edge_index=edge_index, edge_attr=edge_attr, next_x=graph.x[next_node, :1])
                data_list.append(sub_graph)

        print('len of data\t', len(data_list))
        data, slices = self.collate(data_list)
        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])
        return

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
