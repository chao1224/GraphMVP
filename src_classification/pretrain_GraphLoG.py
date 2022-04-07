import argparse
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import GNN
from torch_geometric.data import DataLoader
from torch_geometric.nn import (global_add_pool, global_max_pool,
                                global_mean_pool)

from datasets import MoleculeDataset


class ProjectNet(nn.Module):
    def __init__(self, rep_dim):
        super(ProjectNet, self).__init__()
        self.rep_dim = rep_dim
        self.proj = nn.Sequential(
            nn.Linear(self.rep_dim, self.rep_dim),
            nn.ReLU(),
            nn.Linear(self.rep_dim, self.rep_dim))

    def forward(self, x):
        return self.proj(x)


# Graph pooling functions
def pool_func(x, batch, mode="mean"):
    if mode == "sum":
        return global_add_pool(x, batch)
    elif mode == "mean":
        return global_mean_pool(x, batch)
    elif mode == "max":
        return global_max_pool(x, batch)
    else:
        raise ValueError("mode must be sum or mean or max")

# Mask some nodes in a graph
def mask_nodes(batch, args, num_atom_type=119):
    masked_node_indices = list()
    # select indices of masked nodes
    for i in range(batch.batch[-1] + 1):
        idx = torch.nonzero((batch.batch == i).float()).squeeze(-1)
        num_node = idx.shape[0]
        if args.mask_num == 0:
            sample_size = int(num_node * args.mask_rate + 1)
        else:
            sample_size = min(args.mask_num, int(num_node * 0.5))
        masked_node_idx = random.sample(idx.tolist(), sample_size)
        masked_node_idx.sort()
        masked_node_indices += masked_node_idx

    batch.masked_node_indices = torch.tensor(masked_node_indices)

    # mask nodes' features
    for node_idx in masked_node_indices:
        batch.x[node_idx] = torch.tensor([num_atom_type, 0])

    return batch


# InfoNCE loss within a graph, with and without masking
def intra_NCE_loss(node_reps, node_mask_reps, batch, tau=0.1, epsilon=1e-6):
    node_reps_norm = torch.norm(node_reps, dim=1).unsqueeze(-1)
    node_mask_reps_norm = torch.norm(node_mask_reps, dim=1).unsqueeze(-1)
    sim = torch.mm(node_reps, node_mask_reps.t()) / \
          (torch.mm(node_reps_norm, node_mask_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.stack([(batch.batch == i).float()
                        for i in batch.batch.tolist()], dim=1)
    exp_sim_mask = exp_sim * mask
    exp_sim_all = torch.index_select(exp_sim_mask, 1, batch.masked_node_indices)
    exp_sim_positive = torch.index_select(exp_sim_all, 0, batch.masked_node_indices)
    positive_ratio = exp_sim_positive.sum(0) / (exp_sim_all.sum(0) + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / batch.masked_node_indices.shape[0]
    mask_select = torch.index_select(mask, 1, batch.masked_node_indices)
    thr = 1. / mask_select.sum(0)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# InfoNCE loss across different graphs
def inter_NCE_loss(graph_reps, graph_mask_reps, device, tau=0.1, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    graph_mask_reps_norm = torch.norm(graph_mask_reps, dim=1).unsqueeze(-1)
    sim = torch.mm(graph_reps, graph_mask_reps.t()) / \
          (torch.mm(graph_reps_norm, graph_mask_reps_norm.t()) + epsilon)
    exp_sim = torch.exp(sim / tau)

    mask = torch.eye(graph_reps.shape[0]).to(device)
    positive = (exp_sim * mask).sum(0)
    negative = (exp_sim * (1 - mask)).sum(0)
    positive_ratio = positive / (positive + negative + epsilon)

    NCE_loss = -torch.log(positive_ratio).sum() / graph_reps.shape[0]
    thr = 1. / ((1 - mask).sum(0) + 1.)
    correct_cnt = (positive_ratio > thr).float().sum()

    return NCE_loss, correct_cnt


# InfoNCE loss between graphs and prototypes
def proto_NCE_loss(graph_reps, graph_mask_reps, proto_list,
                   proto_connection, tau=0.1, decay_ratio=0.7, epsilon=1e-6):
    # similarity for original and masked graphs
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    graph_mask_reps_norm = torch.norm(graph_mask_reps, dim=1).unsqueeze(-1)
    mask_list = []
    exp_sim_list = []
    exp_sim_mask_list = []

    for i in range(len(proto_list) - 1, -1, -1):
        proto = proto_list[i]
        proto_norm = torch.norm(proto, dim=1).unsqueeze(-1)

        sim = torch.mm(graph_reps, proto.t()) / (
                torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)
        exp_sim = torch.exp(sim / tau)
        sim_mask = torch.mm(graph_mask_reps, proto.t()) / (
                torch.mm(graph_mask_reps_norm, proto_norm.t()) + epsilon)
        exp_sim_mask = torch.exp(sim_mask / tau)
        if i != (len(proto_list) - 1):
            exp_sim_last = exp_sim_list[-1]
            idx_last = torch.argmax(exp_sim_last, dim=1).unsqueeze(-1)
            connection = proto_connection[i]
            connection_mask = (connection.unsqueeze(0) == idx_last.float()).float()
            exp_sim = exp_sim * connection_mask
            exp_sim_mask = exp_sim_mask * connection_mask

        mask = (exp_sim == exp_sim.max(1)[0].unsqueeze(-1)).float()

        exp_sim_mask_list.append(exp_sim_mask)
        exp_sim_list.append(exp_sim)
        mask_list.append(mask)

    # define InfoNCE loss
    NCE_loss = 0

    for i in range(len(proto_list)):
        exp_sim_mask = exp_sim_mask_list[i]
        mask = mask_list[i]

        positive = exp_sim_mask * mask
        negative = exp_sim_mask * (1 - mask)
        positive_ratio = positive.sum(1) / (positive.sum(1) + negative.sum(1) + epsilon)
        NCE_loss += -torch.log(positive_ratio).mean()

    # update prototypes
    mask_last = mask_list[-1]
    cnt = mask_last.sum(0)
    batch_cnt = mask_last.t() / (cnt.unsqueeze(-1) + epsilon)
    batch_mean = torch.mm(batch_cnt, graph_reps)
    proto_list[0].data = proto_list[0].data * (cnt == 0).float().unsqueeze(-1).data + \
                         (proto_list[0].data * decay_ratio +
                          batch_mean.data * (1 - decay_ratio)) \
                         * (cnt != 0).float().unsqueeze(-1).data

    for i in range(1, len(proto_list)):
        proto = proto_list[i]
        proto_last = proto_list[i - 1]
        connection = proto_connection[i - 1]
        connection_mask = torch.stack([(connection == j).float()
                                       for j in range(proto.shape[0])], dim=0)
        connection_cnt = connection_mask.sum(1).unsqueeze(-1)
        connection_weight = connection_mask / (connection_cnt + epsilon)
        proto_list[i].data = torch.mm(connection_weight, proto_last).data

    return NCE_loss, proto_list


# Update prototypes with batch information
def update_proto_lowest(graph_reps, proto, proto_state, decay_ratio=0.7, epsilon=1e-6):
    graph_reps_norm = torch.norm(graph_reps, dim=1).unsqueeze(-1)
    proto_norm = torch.norm(proto, dim=1).unsqueeze(-1)
    sim = torch.mm(graph_reps, proto.t()) / (
            torch.mm(graph_reps_norm, proto_norm.t()) + epsilon)

    # update states of prototypes
    mask = (sim == sim.max(1)[0].unsqueeze(-1)).float()
    cnt = mask.sum(0)
    proto_state.data = proto_state.data + (cnt != 0).float().data - \
                       proto_state.data * (cnt != 0).float().data

    # update prototypes
    batch_cnt = mask.t() / (cnt.unsqueeze(-1) + epsilon)
    batch_mean = torch.mm(batch_cnt, graph_reps)
    proto.data = proto.data * (cnt == 0).float().unsqueeze(-1).data + \
                 (proto.data * decay_ratio + batch_mean.data * (1 - decay_ratio)) \
                 * (cnt != 0).float().unsqueeze(-1).data

    return proto, proto_state


# Initialize prototypes and their state
def init_proto_lowest(args, model, proj, loader, device, proto, proto_state):
    model.eval()
    proj.eval()

    for step, batch in enumerate(loader):
        batch = batch.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)

        # feature projection
        graph_reps_proj = proj(graph_reps)

        # update prototypes
        proto, proto_state = update_proto_lowest(
            graph_reps_proj, proto, proto_state, decay_ratio=args.decay_ratio)

    idx = torch.nonzero(proto_state).squeeze(-1)
    proto_selected = torch.index_select(proto, 0, idx)

    return proto_selected, proto_state


# Initialize prototypes and their state
def init_proto(args, proto_, proto, proto_state, device, num_iter=20):
    proto_connection = torch.zeros(proto_.shape[0]).to(device)

    for iter in range(num_iter):
        for i in range(proto_.shape[0]):
            # update winner
            sim = torch.mm(proto, proto_[i, :].unsqueeze(-1)).squeeze(-1)
            idx = torch.argmax(sim)
            if iter == (num_iter - 1):
                proto_state[idx] = 1
            proto_connection[i] = idx
            proto.data[idx, :] = proto.data[idx, :] * args.decay_ratio + \
                                 proto_.data[i, :] * (1 - args.decay_ratio)

            # penalize rival
            sim[idx] = 0
            rival_idx = torch.argmax(sim)
            proto.data[rival_idx, :] = proto.data[rival_idx, :] * (2 - args.decay_ratio) - \
                                       proto_.data[i, :] * (1 - args.decay_ratio)

    indices = torch.nonzero(proto_state).squeeze(-1)
    proto_selected = torch.index_select(proto, 0, indices)
    for i in range(indices.shape[0]):
        idx = indices[i].float()
        idx_connection = torch.nonzero((proto_connection == idx).float()).squeeze(-1)
        proto_connection[idx_connection] = i

    return proto_selected, proto_state, proto_connection


# For one epoch pretraining
def pretrain(args, model, proj, loader, optimizer, device):
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(loader):
        batch_mask = copy.deepcopy(batch)
        batch_mask = mask_nodes(batch_mask, args)
        batch, batch_mask = batch.to(device), batch_mask.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_mask_reps = model(batch_mask.x, batch_mask.edge_index, batch_mask.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_mask_reps = pool_func(node_mask_reps, batch_mask.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_mask_reps_proj = proj(node_mask_reps)
        graph_reps_proj = proj(graph_reps)
        graph_mask_reps_proj = proj(graph_mask_reps)

        # InfoNCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(node_reps_proj, node_mask_reps_proj,
                                                       batch_mask, tau=args.tau)
        NCE_loss_inter, correct_inter = inter_NCE_loss(graph_reps_proj, graph_mask_reps_proj,
                                                       device, tau=args.tau)

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_mask.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        optimizer.zero_grad()
        NCE_loss = args.alpha * NCE_loss_intra + args.beta * NCE_loss_inter
        NCE_loss.backward()
        optimizer.step()

        if (step + 1) % args.disp_interval == 0:
            print('iteration: %5d, intra NCE loss: %.3f, intra acc: %.3f, '
                  'inter NCE loss: %.3f, inter acc: %.3f' % (
                step + 1,
                NCE_loss_intra.item(),
                float(correct_intra_cnt) / float(total_intra_cnt),
                NCE_loss_inter.item(),
                float(correct_inter_cnt) / float(total_inter_cnt)))

    return NCE_loss_intra_cnt / step, \
           float(correct_intra_cnt) / float(total_intra_cnt), \
           NCE_loss_inter_cnt / step, \
           float(correct_inter_cnt) / float(total_inter_cnt)


# For every epoch training (fine-tuning)
def train(args, model, proj, loader, optimizer, device, proto, proto_connection):
    model.train()
    proj.train()

    NCE_loss_intra_cnt = 0
    NCE_loss_inter_cnt = 0
    NCE_loss_proto_cnt = 0
    correct_intra_cnt = 0
    correct_inter_cnt = 0
    total_intra_cnt = 0
    total_inter_cnt = 0

    for step, batch in enumerate(loader):
        batch_mask = copy.deepcopy(batch)
        batch_mask = mask_nodes(batch_mask, args)
        batch, batch_mask = batch.to(device), batch_mask.to(device)

        # get node and graph representations
        node_reps = model(batch.x, batch.edge_index, batch.edge_attr)
        node_mask_reps = model(batch_mask.x, batch_mask.edge_index, batch_mask.edge_attr)
        graph_reps = pool_func(node_reps, batch.batch, mode=args.graph_pooling)
        graph_mask_reps = pool_func(node_mask_reps, batch_mask.batch, mode=args.graph_pooling)

        # feature projection
        node_reps_proj = proj(node_reps)
        node_mask_reps_proj = proj(node_mask_reps)
        graph_reps_proj = proj(graph_reps)
        graph_mask_reps_proj = proj(graph_mask_reps)

        # InfoNCE loss
        NCE_loss_intra, correct_intra = intra_NCE_loss(node_reps_proj, node_mask_reps_proj,
                                                       batch_mask, tau=args.tau)
        NCE_loss_inter, correct_inter = inter_NCE_loss(graph_reps_proj, graph_mask_reps_proj,
                                                       device, tau=args.tau)
        NCE_loss_proto, proto = proto_NCE_loss(
            graph_reps_proj, graph_mask_reps_proj, proto,
            proto_connection, tau=args.tau, decay_ratio=args.decay_ratio)

        NCE_loss_intra_cnt += NCE_loss_intra.item()
        NCE_loss_inter_cnt += NCE_loss_inter.item()
        NCE_loss_proto_cnt += NCE_loss_proto.item()
        correct_intra_cnt += correct_intra
        correct_inter_cnt += correct_inter
        total_intra_cnt += batch_mask.masked_node_indices.shape[0]
        total_inter_cnt += graph_reps.shape[0]

        # optimization
        NCE_loss = args.alpha * NCE_loss_intra + \
                   args.beta * NCE_loss_inter + \
                   args.gamma * NCE_loss_proto
        optimizer.zero_grad()
        NCE_loss.backward()
        optimizer.step()

        if (step + 1) % args.disp_interval == 0:
            print(
                'iteration: %5d, intra NCE loss: %.3f, intra acc: %.3f, '
                'inter NCE loss: %.3f, inter acc: %.3f' % (
                    step + 1,
                    NCE_loss_intra.item(),
                    float(correct_intra_cnt) / float(total_intra_cnt),
                    NCE_loss_inter.item(),
                    float(correct_inter_cnt) / float(total_inter_cnt)))

            template = 'iteration: %5d, proto NCE loss: %.3f'
            value_list = [step + 1, NCE_loss_proto.item()]
            for i in range(args.hierarchy):
                template += (', active num ' + str(i + 1) + ': %5d')
                value_list.append(proto[i].shape[0])
            print(template % tuple(value_list))

    return NCE_loss_intra_cnt / step, \
           float(correct_intra_cnt) / float(total_intra_cnt), \
           NCE_loss_inter_cnt / step, \
           float(correct_inter_cnt) / float(total_inter_cnt), \
           NCE_loss_proto_cnt / step, proto


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GraphLoG for GNN pre-training')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay in Adam (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--mask_rate', type=float, default=0.3,
                        help='dropout ratio (default: 0.15)')
    parser.add_argument('--mask_num', type=int, default=0,
                        help='the number of masked nodes (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features are combined across layers.')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max)')
    parser.add_argument('--dataset', type=str, default='',
                        help='root directory of dataset for pretraining')
    parser.add_argument('--output_model_dir', type=str, default='',
                        help='filename to output the model')
    parser.add_argument('--gnn_type', type=str, default="gin",
                        help='GNN model (default: GIN)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for splitting dataset (default: 0)')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='number of workers for dataset loading (default: 1)')
    parser.add_argument('--tau', type=float, default=0.04,
                        help='the temperature parameter for softmax (default: 0.04)')
    parser.add_argument('--decay_ratio', type=float, default=0.95,
                        help='the decay ratio for moving average (default: 0.95)')
    parser.add_argument('--num_proto', type=int, default=100,
                        help='the number of initial prototypes (default: 100)')
    parser.add_argument('--hierarchy', type=int, default=3,
                        help='the number of hierarchy in prototypes (default: 3)')
    parser.add_argument('--alpha', type=float, default=1,
                        help='the weight of intra-graph InfoNCE loss (default: 1)')
    parser.add_argument('--beta', type=float, default=1,
                        help='the weight of inter-graph InfoNCE loss (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='the weight of prototype InfoNCE loss (default: 0.1)')
    parser.add_argument('--disp_interval', type=int, default=10,
                        help='the display interval (default: 10)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # set up dataset and transform function.
    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset('../datasets/{}/'.format(args.dataset), dataset=args.dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers)

    # set up pretraining models and feature projector
    print("num of GNN layers: %d" % args.num_layer)
    model = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
                gnn_type=args.gnn_type).to(device)
    if args.JK == 'concat':
        proj = ProjectNet((args.num_layer + 1) * args.emb_dim).to(device)
    else:
        proj = ProjectNet(args.emb_dim).to(device)

    # set up optimizer
    model_param_group = [{"params": model.parameters(), "lr": args.lr},
                         {"params": proj.parameters(),  "lr": args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    # initialize prototypes and their state
    if args.JK == 'concat':
        proto = [torch.rand((args.num_proto, (args.num_layer + 1) * args.emb_dim)).to(device)
                 for _ in range(args.hierarchy)]
    else:
        proto = [torch.rand((args.num_proto, args.emb_dim)).to(device)
                 for _ in range(args.hierarchy)]
    proto_state = [torch.zeros(args.num_proto).to(device)
                   for _ in range(args.hierarchy)]
    proto_connection = []

    # pre-training with only intra and inter InfoNCE losses
    intra_loss, intra_acc, inter_loss, inter_acc = pretrain(
        args, model, proj, loader, optimizer, device)
    print(intra_loss, intra_acc, inter_loss, inter_acc)

    # initialize prototypes and their state according to pretrained representations
    print("Initialize prototypes: layer 1")
    tmp_proto, tmp_proto_state = init_proto_lowest(
        args, model, proj, loader, device, proto[0], proto_state[0])
    proto[0], proto_state[0] = tmp_proto, tmp_proto_state

    for i in range(1, args.hierarchy):
        print("Initialize prototypes: layer ", i + 1)
        tmp_proto, tmp_proto_state, tmp_proto_connection = init_proto(
            args, proto[i - 1], proto[i], proto_state[i], device)
        proto[i] = tmp_proto
        proto_state[i] = tmp_proto_state
        proto_connection.append(tmp_proto_connection)

    for epoch in range(1, args.epochs + 1):
        print("=== epoch: %3d ===" % epoch)
        intra_loss, intra_acc, inter_loss, inter_acc, proto_loss, proto = train(
            args, model, proj, loader, optimizer, device, proto, proto_connection)
        print(intra_loss, intra_acc, inter_loss, inter_acc, proto_loss)
        print("\n\n")

    if not args.output_model_dir == '':
        torch.save(model.state_dict(), args.output_model_dir + '_model.pth')
