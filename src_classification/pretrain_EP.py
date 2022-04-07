import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from dataloader import DataLoaderAE
from models import GNN
from torch_geometric.nn import global_mean_pool
from util import NegativeEdge

from datasets import MoleculeDataset


def do_EdgePred(node_repr, batch, criterion=nn.BCEWithLogitsLoss()):

    # positive/negative scores -> inner product of node features
    positive_score = torch.sum(node_repr[batch.edge_index[0, ::2]] *
                               node_repr[batch.edge_index[1, ::2]], dim=1)
    negative_score = torch.sum(node_repr[batch.negative_edge_index[0]] *
                               node_repr[batch.negative_edge_index[1]], dim=1)

    edgepred_loss = criterion(positive_score, torch.ones_like(positive_score)) + \
                    criterion(negative_score, torch.zeros_like(negative_score))
    edgepred_acc = (torch.sum(positive_score > 0) +
                    torch.sum(negative_score < 0)).to(torch.float32) / \
                   float(2 * len(positive_score))
    edgepred_acc = edgepred_acc.detach().cpu().item()

    return edgepred_loss, edgepred_acc


def train(molecule_model, device, loader, optimizer,
          criterion=nn.BCEWithLogitsLoss()):

    # Train for one epoch
    molecule_model.train()
    start_time = time.time()
    edgepred_loss_accum, edgepred_acc_accum = 0, 0

    for step, batch in enumerate(loader):

        batch = batch.to(device)

        node_repr = molecule_model(batch.x, batch.edge_index, batch.edge_attr)
        edgepred_loss, edgepred_acc = do_EdgePred(
            node_repr=node_repr, batch=batch, criterion=criterion)
        edgepred_loss_accum += edgepred_loss.detach().cpu().item()
        edgepred_acc_accum += edgepred_acc
        ssl_loss = edgepred_loss

        optimizer.zero_grad()
        ssl_loss.backward()
        optimizer.step()

    print('EP Loss: {:.5f}\tEP Acc: {:.5f}\tTime: {:.5f}'.format(
        edgepred_loss_accum / len(loader),
        edgepred_acc_accum / len(loader),
        time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset('../datasets/{}/'.format(args.dataset), dataset=args.dataset, transform=NegativeEdge())
    loader = DataLoaderAE(dataset, batch_size=args.batch_size,
                          shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(args.num_layer, args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    model_param_group = [{'params': molecule_model.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(molecule_model, device, loader, optimizer)

    if not args.output_model_dir == '':
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')
        saver_dict = {'model': molecule_model.state_dict()}
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
