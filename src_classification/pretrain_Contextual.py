import time
from os.path import join

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
from util import get_num_task

from datasets import MoleculeContextualDataset


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item())/len(pred)


def do_Contextual(batch, criterion, node_repr, atom_vocab_model):
    target = batch.atom_vocab_label
    node_pred = atom_vocab_model(node_repr)
    loss = criterion(node_pred, target)
    acc = compute_accuracy(node_pred, target)
    return loss, acc


def train(device, loader, optimizer):
    start = time.time()
    molecule_model.train()
    atom_vocab_model.train()

    contextual_loss_accum, contextual_acc_accum = 0, 0
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        node_repr = molecule_model(batch.x, batch.edge_index, batch.edge_attr)
        
        contextual_loss, contextual_acc = do_Contextual(batch, criterion, node_repr, atom_vocab_model)
        contextual_loss_accum += contextual_loss.detach().cpu().item()
        contextual_acc_accum += contextual_acc
        loss = contextual_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('Contextual Loss: {:.5f}\tContextual Acc: {:.5f}\tTime: {:.5f}'.format(
        contextual_loss_accum / len(loader),
        contextual_acc_accum / len(loader),
        time.time() - start))
    return


if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    assert 'GEOM' in args.dataset
    dataset_folder = '../datasets/'
    dataset = MoleculeContextualDataset(dataset_folder + args.dataset, dataset=args.dataset)
    print(dataset)
    
    atom_vocab = dataset.atom_vocab
    atom_vocab_size = len(atom_vocab)
    print('atom_vocab\t', len(atom_vocab), atom_vocab_size)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type).to(device)
    atom_vocab_model = nn.Linear(args.emb_dim, atom_vocab_size).to(device)

    # set up optimizer
    # different learning rates for different parts of GNN
    model_param_group = [{'params': molecule_model.parameters()},
                         {'params': atom_vocab_model.parameters(), 'lr': args.lr * args.lr_scale}]
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.CrossEntropyLoss()
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0

    print('\nStart pre-training Contextual')
    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(device, loader, optimizer)

    if args.output_model_dir is not '':
        print('saving to {}'.format(args.output_model_dir + '_model.pth'))
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'atom_vocab_model': atom_vocab_model.state_dict(),
        }
        torch.save(saved_model_dict, args.output_model_dir + '_model_complete.pth')
