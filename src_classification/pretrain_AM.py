import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from dataloader import DataLoaderMasking
from models import GNN
from torch_geometric.nn import global_mean_pool
from util import MaskAtom

from datasets import MoleculeDataset


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def do_AttrMasking(batch, criterion, node_repr, molecule_atom_masking_model):
    target = batch.mask_node_label[:, 0]
    node_pred = molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
    attributemask_loss = criterion(node_pred.double(), target)
    attributemask_acc = compute_accuracy(node_pred, target)
    return attributemask_loss, attributemask_acc


def train(device, loader, optimizer):

    start = time.time()
    molecule_model.train()
    molecule_atom_masking_model.train()
    attributemask_loss_accum, attributemask_acc_accum = 0, 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        node_repr = molecule_model(batch.masked_x, batch.edge_index, batch.edge_attr)

        attributemask_loss, attributemask_acc = do_AttrMasking(
            batch=batch, criterion=criterion, node_repr=node_repr,
            molecule_atom_masking_model=molecule_atom_masking_model)

        attributemask_loss_accum += attributemask_loss.detach().cpu().item()
        attributemask_acc_accum += attributemask_acc
        loss = attributemask_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('AM Loss: {:.5f}\tAM Acc: {:.5f}\tTime: {:.5f}'.format(
        attributemask_loss_accum / len(loader),
        attributemask_acc_accum / len(loader),
        time.time() - start))
    return


if __name__ == '__main__':

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset(
            '../datasets/{}/'.format(args.dataset), dataset=args.dataset,
            transform=MaskAtom(num_atom_type=119, num_edge_type=5,
                               mask_rate=args.mask_rate, mask_edge=args.mask_edge))
    loader = DataLoaderMasking(dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(args.num_layer, args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    molecule_atom_masking_model = torch.nn.Linear(args.emb_dim, 119).to(device)

    model_param_group = [{'params': molecule_model.parameters(), 'lr': args.lr},
                         {'params': molecule_atom_masking_model.parameters(), 'lr': args.lr}]

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(device, loader, optimizer)

    if not args.output_model_dir == '':
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')

        saver_dict = {'model':                       molecule_model.state_dict(),
                      'molecule_atom_masking_model': molecule_atom_masking_model.state_dict()}

        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
