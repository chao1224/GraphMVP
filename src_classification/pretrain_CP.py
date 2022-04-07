import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from dataloader import DataLoaderSubstructContext
from models import GNN
from torch_geometric.nn import global_mean_pool
from util import ExtractSubstructureContextPair, cycle_index

from datasets import MoleculeDataset


def do_ContextPred(batch, criterion, args, molecule_substruct_model,
                   molecule_context_model, molecule_readout_func):

    # creating substructure representation
    substruct_repr = molecule_substruct_model(
        batch.x_substruct, batch.edge_index_substruct,
        batch.edge_attr_substruct)[batch.center_substruct_idx]

    # creating context representations
    overlapped_node_repr = molecule_context_model(
        batch.x_context, batch.edge_index_context,
        batch.edge_attr_context)[batch.overlap_context_substruct_idx]

    # positive context representation
    # readout -> global_mean_pool by default
    context_repr = molecule_readout_func(overlapped_node_repr,
                                         batch.batch_overlapped_context)

    # negative contexts are obtained by shifting
    # the indices of context embeddings
    neg_context_repr = torch.cat(
        [context_repr[cycle_index(len(context_repr), i + 1)]
         for i in range(args.contextpred_neg_samples)], dim=0)

    num_neg = args.contextpred_neg_samples
    pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
    pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

    loss_pos = criterion(pred_pos.double(),
                         torch.ones(len(pred_pos)).to(pred_pos.device).double())
    loss_neg = criterion(pred_neg.double(),
                         torch.zeros(len(pred_neg)).to(pred_neg.device).double())

    contextpred_loss = loss_pos + num_neg * loss_neg

    num_pred = len(pred_pos) + len(pred_neg)
    contextpred_acc = (torch.sum(pred_pos > 0).float() +
                       torch.sum(pred_neg < 0).float()) / num_pred
    contextpred_acc = contextpred_acc.detach().cpu().item()

    return contextpred_loss, contextpred_acc


def train(args, device, loader, optimizer):

    start_time = time.time()
    molecule_context_model.train()
    molecule_substruct_model.train()
    contextpred_loss_accum, contextpred_acc_accum = 0, 0

    for step, batch in enumerate(loader):

        batch = batch.to(device)
        contextpred_loss, contextpred_acc = do_ContextPred(
            batch=batch, criterion=criterion, args=args,
            molecule_substruct_model=molecule_substruct_model,
            molecule_context_model=molecule_context_model,
            molecule_readout_func=molecule_readout_func)

        contextpred_loss_accum += contextpred_loss.detach().cpu().item()
        contextpred_acc_accum += contextpred_acc
        ssl_loss = contextpred_loss
        optimizer.zero_grad()
        ssl_loss.backward()
        optimizer.step()

    print('CP Loss: {:.5f}\tCP Acc: {:.5f}\tTime: {:.3f}'.format(
        contextpred_loss_accum / len(loader),
        contextpred_acc_accum / len(loader),
        time.time() - start_time))

    return


if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    l1 = args.num_layer - 1
    l2 = l1 + args.csize
    print('num layer: %d l1: %d l2: %d' % (args.num_layer, l1, l2))

    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset(
            '../datasets/{}/'.format(args.dataset), dataset=args.dataset,
            transform=ExtractSubstructureContextPair(args.num_layer, l1, l2))
    loader = DataLoaderSubstructContext(dataset, batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.num_workers)

    ''' === set up model, mainly used in do_ContextPred() === '''
    molecule_substruct_model = GNN(
        args.num_layer, args.emb_dim, JK=args.JK,
        drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_context_model = GNN(
        int(l2 - l1), args.emb_dim, JK=args.JK,
        drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)

    ''' === set up loss and optimiser === '''
    criterion = nn.BCEWithLogitsLoss()
    molecule_readout_func = global_mean_pool
    model_param_group = [{'params': molecule_substruct_model.parameters(), 'lr': args.lr},
                         {'params': molecule_context_model.parameters(), 'lr': args.lr}]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, device, loader, optimizer)

    if not args.output_model_dir == '':
        torch.save(molecule_substruct_model.state_dict(),
                   args.output_model_dir + '_model.pth')

        saver_dict = {
            'molecule_substruct_model': molecule_substruct_model.state_dict(),
            'molecule_context_model':   molecule_context_model.state_dict()}

        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
