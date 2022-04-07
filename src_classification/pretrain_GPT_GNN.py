import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models import GNN
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool

from datasets import MoleculeDataset, MoleculeDatasetGPT


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def train(device, loader, optimizer):
    start = time.time()
    molecule_model.train()
    node_pred_model.train()
    gpt_loss_accum, gpt_acc_accum = 0, 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        node_repr = molecule_model(batch.x, batch.edge_index, batch.edge_attr)
        graph_repr = molecule_readout_func(node_repr, batch.batch)
        node_pred = node_pred_model(graph_repr)
        target = batch.next_x

        gpt_loss = criterion(node_pred.double(), target)
        gpt_acc = compute_accuracy(node_pred, target)

        gpt_loss_accum += gpt_loss.detach().cpu().item()
        gpt_acc_accum += gpt_acc
        loss = gpt_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('GPT Loss: {:.5f}\tGPT Acc: {:.5f}\tTime: {:.5f}'.format(
        gpt_loss_accum / len(loader), gpt_acc_accum / len(loader), time.time() - start))
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
        molecule_dataset = MoleculeDataset('../datasets/{}/'.format(args.dataset), dataset=args.dataset)
    molecule_gpt_dataset = MoleculeDatasetGPT(molecule_dataset)
    loader = DataLoader(molecule_gpt_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type).to(device)
    node_pred_model = nn.Linear(args.emb_dim, 120).to(device)

    model_param_group = [
        {'params': molecule_model.parameters(), 'lr': args.lr},
        {'params': node_pred_model.parameters(), 'lr': args.lr},
    ]

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    molecule_readout_func = global_mean_pool
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(device, loader, optimizer)

    if not args.output_model_dir == '':
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')
        saver_dict = {
            'model': molecule_model.state_dict(),
            'node_pred_model': node_pred_model.state_dict(),
        }
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
