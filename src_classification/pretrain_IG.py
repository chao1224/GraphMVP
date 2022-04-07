import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models import GNN, Discriminator
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from util import cycle_index

from datasets import MoleculeDataset


def do_InfoGraph(node_repr, molecule_repr, batch,
                 criterion, infograph_discriminator_SSL_model):

    summary_repr = torch.sigmoid(molecule_repr)
    positive_expanded_summary_repr = summary_repr[batch.batch]
    shifted_summary_repr = summary_repr[cycle_index(len(summary_repr), 1)]
    negative_expanded_summary_repr = shifted_summary_repr[batch.batch]

    positive_score = infograph_discriminator_SSL_model(
        node_repr, positive_expanded_summary_repr)
    negative_score = infograph_discriminator_SSL_model(
        node_repr, negative_expanded_summary_repr)
    infograph_loss = criterion(positive_score, torch.ones_like(positive_score)) + \
                     criterion(negative_score, torch.zeros_like(negative_score))

    num_sample = float(2 * len(positive_score))
    infograph_acc = (torch.sum(positive_score > 0) +
                     torch.sum(negative_score < 0)).to(torch.float32) / num_sample
    infograph_acc = infograph_acc.detach().cpu().item()

    return infograph_loss, infograph_acc


def train(molecule_model, device, loader, optimizer):

    start = time.time()
    molecule_model.train()
    infograph_loss_accum, infograph_acc_accum = 0, 0

    for step, batch in enumerate(loader):

        batch = batch.to(device)
        node_repr = molecule_model(batch.x, batch.edge_index, batch.edge_attr)
        molecule_repr = molecule_readout_func(node_repr, batch.batch)

        infograph_loss, infograph_acc = do_InfoGraph(
            node_repr=node_repr, batch=batch,
            molecule_repr=molecule_repr, criterion=criterion,
            infograph_discriminator_SSL_model=infograph_discriminator_SSL_model)

        infograph_loss_accum += infograph_loss.detach().cpu().item()
        infograph_acc_accum += infograph_acc
        ssl_loss = infograph_loss
        optimizer.zero_grad()
        ssl_loss.backward()
        optimizer.step()

    print('IG Loss: {:.5f}\tIG Acc: {:.5f}\tTime: {:.3f}'.format(
        infograph_loss_accum / len(loader),
        infograph_acc_accum / len(loader),
        time.time() - start))
    return


if __name__ == '__main__':

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset('../datasets/{}/'.format(args.dataset), dataset=args.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type).to(device)
    infograph_discriminator_SSL_model = Discriminator(args.emb_dim).to(device)

    model_param_group = [{'params': molecule_model.parameters(), 'lr': args.lr},
                         {'params': infograph_discriminator_SSL_model.parameters(),
                          'lr': args.lr}]

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    molecule_readout_func = global_mean_pool
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(molecule_model, device, loader, optimizer)

    if not args.output_model_dir == '':
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')

        saver_dict = {'model':                             molecule_model.state_dict(),
                      'infograph_discriminator_SSL_model': infograph_discriminator_SSL_model.state_dict()}

        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
