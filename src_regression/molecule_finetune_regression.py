import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, '../src_classification')
from os.path import join

from config import args
from datasets_complete_feature import MoleculeDatasetComplete
from models_complete_feature import GNN_graphpredComplete, GNNComplete
from sklearn.metrics import mean_absolute_error, mean_squared_error
from splitters import random_scaffold_split, random_split, scaffold_split
from torch_geometric.data import DataLoader


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
        y = batch.y.squeeze()

        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze(1)
 
        true = batch.y.view(pred.shape)
        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    num_tasks = 1
    dataset_folder = '../datasets/molecule_datasets_regression/'
    dataset_folder = os.path.join(dataset_folder, args.dataset)
    dataset = MoleculeDatasetComplete(dataset_folder, dataset=args.dataset)
    print('dataset_folder:', dataset_folder)
    print(dataset)

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # set up model
    molecule_model = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GNN_graphpredComplete(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file)
    model.to(device)
    print(model)

    model_param_group = [
        {'params': model.molecule_model.parameters()},
        {'params': model.graph_pred_linear.parameters(), 'lr': args.lr * args.lr_scale}
    ]
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # reg_criterion = torch.nn.L1Loss()
    reg_criterion = torch.nn.MSELoss()

    train_result_list, val_result_list, test_result_list = [], [], []
    # metric_list = ['RMSE', 'MAE', 'R2']
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval(model, device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred = eval(model, device, val_loader)
        test_result, test_target, test_pred = eval(model, device, test_loader)

        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()

        if val_result['RMSE'] < best_val_rmse:
            best_val_rmse = val_result['RMSE']
            best_val_idx = epoch - 1
            if not args.output_model_dir == '':
                output_model_path = join(args.output_model_dir, 'model_best.pth')
                saved_model_dict = {
                    'molecule_model': molecule_model.state_dict(),
                    'model': model.state_dict()
                }
                torch.save(saved_model_dict, output_model_path)

                filename = join(args.output_model_dir, 'evaluation_best.pth')
                np.savez(filename, val_target=val_target, val_pred=val_pred,
                         test_target=test_target, test_pred=test_pred)

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
        
    if args.output_model_dir is not '':
        output_model_path = join(args.output_model_dir, 'model_final.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict()
        }
        torch.save(saved_model_dict, output_model_path)
