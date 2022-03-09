import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, '../src_classification')
from datasets_complete_feature import MoleculeProteinDataset
from models import MoleculeProteinModel, ProteinModel
from models_complete_feature import GNNComplete
from util import ci, mse, pearson, rmse, spearman


def train(repurpose_model, device, dataloader, optimizer):
    repurpose_model.train()
    loss_accum = 0
    for step_idx, batch in enumerate(dataloader):
        molecule, protein, label = batch
        molecule = molecule.to(device)
        protein = protein.to(device)
        label = label.to(device)

        pred = repurpose_model(molecule, protein).squeeze()

        optimizer.zero_grad()
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().item()
    print('Loss:\t{}'.format(loss_accum / len(dataloader)))


def predicting(repurpose_model, device, dataloader):
    repurpose_model.eval()
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for batch in dataloader:
            molecule, protein, label = batch
            molecule = molecule.to(device)
            protein = protein.to(device)
            label = label.to(device)
            pred = repurpose_model(molecule, protein).squeeze()

            total_preds.append(pred.detach().cpu())
            total_labels.append(label.detach().cpu())
    total_preds = torch.cat(total_preds, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    return total_labels.numpy(), total_preds.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.)
    parser.add_argument('--graph_pooling', type=str, default='mean')
    parser.add_argument('--JK', type=str, default='last')
    parser.add_argument('--dataset', type=str, default='davis', choices=['davis', 'kiba'])
    parser.add_argument('--gnn_type', type=str, default='gin')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--input_model_file', type=str, default='')
    parser.add_argument('--output_model_file', type=str, default='')
    ########## For protein embedding ##########
    parser.add_argument('--protein_emb_dim', type=int, default=300)
    parser.add_argument('--protein_hidden_dim', type=int, default=300)
    parser.add_argument('--num_features', type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')

    ########## Set up dataset and dataloader ##########
    root = '../datasets/dti_datasets'
    train_val_dataset = MoleculeProteinDataset(root=root, dataset=args.dataset, mode='train')
    train_size = int(0.8 * len(train_val_dataset))
    valid_size = len(train_val_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, valid_size])
    test_dataset = MoleculeProteinDataset(root=root, dataset=args.dataset, mode='test')
    print('size of train: {}\tval: {}\ttest: {}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    ########## Set up model ##########
    molecule_model = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    ########## Load pre-trained model ##########
    if not args.input_model_file == '':
        print('========= Loading from {}'.format(args.input_model_file))
        molecule_model.load_state_dict(torch.load(args.input_model_file))
    protein_model = ProteinModel(
        emb_dim=args.protein_emb_dim, num_features=args.num_features, output_dim=args.protein_hidden_dim)
    repurpose_model = MoleculeProteinModel(
        molecule_model, protein_model,
        molecule_emb_dim=args.emb_dim, protein_emb_dim=args.protein_hidden_dim).to(device)
    print('repurpose model\n', repurpose_model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(repurpose_model.parameters(), lr=args.learning_rate)

    best_repurpose_model = None
    best_mse = 1000
    best_epoch = 0

    for epoch in range(1, 1+args.epochs):
        start_time = time.time()
        print('Start training at epoch: {}'.format(epoch))
        train(repurpose_model, device, train_dataloader, optimizer)

        G, P = predicting(repurpose_model, device, valid_dataloader)
        current_mse = mse(G, P)
        print('MSE:\t{}'.format(current_mse))
        if current_mse < best_mse:
            best_repurpose_model = copy.deepcopy(repurpose_model)
            best_mse = current_mse
            best_epoch = epoch
            print('MSE improved at epoch {}\tbest MSE: {}'.format(best_epoch, best_mse))
        else:
            print('No improvement since epoch {}\tbest MSE: {}'.format(best_epoch, best_mse))
        print('Took {:.5f}s.'.format(time.time() - start_time))
        print()

    start_time = time.time()
    print('Last epoch: {}'.format(args.epochs))
    G, P = predicting(repurpose_model, device, test_dataloader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    print('RMSE: {}\tMSE: {}\tPearson: {}\tSpearman: {}\tCI: {}'.format(ret[0], ret[1], ret[2], ret[3], ret[4]))
    print('Took {:.5f}s.'.format(time.time() - start_time))

    start_time = time.time()
    print('Best epoch: {}'.format(best_epoch))
    G, P = predicting(best_repurpose_model, device, test_dataloader)
    ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
    print('RMSE: {}\tMSE: {}\tPearson: {}\tSpearman: {}\tCI: {}'.format(ret[0], ret[1], ret[2], ret[3], ret[4]))
    print('Took {:.5f}s.'.format(time.time() - start_time))

    if not args.output_model_file == '':
        torch.save({
            'repurpose_model': repurpose_model.state_dict(),
            'best_repurpose_model': best_repurpose_model.state_dict()
        }, args.output_model_file + '.pth')
