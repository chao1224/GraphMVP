import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from models import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader

from datasets import RDKIT_PROPS, MoleculeMotifDataset


def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).double()

        loss = criterion(pred.double(), y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
    return total_loss / len(loader)


def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            true = batch.y.view(pred.shape)
        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i] == 1) > 0:
            roc_list.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
    return sum(roc_list) / len(roc_list), y_true, y_scores


if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = len(RDKIT_PROPS)
    assert 'GEOM' in args.dataset
    dataset_folder = '../datasets/'
    dataset = MoleculeMotifDataset(dataset_folder + args.dataset, dataset=args.dataset)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim,
                         JK=args.JK, drop_ratio=args.dropout_ratio,
                         gnn_type=args.gnn_type)
    model = GNN_graphpred(args=args, num_tasks=num_tasks, molecule_model=molecule_model)
    if not args.input_model_file == '':
        model.from_pretrained(args.input_model_file)
    model.to(device)

    # set up optimizer
    # different learning rates for different parts of GNN
    model_param_group = [{'params': model.molecule_model.parameters()},
                         {'params': model.graph_pred_linear.parameters(),
                          'lr': args.lr * args.lr_scale}]
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss()
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0

    print('\nStart pre-training Motif')
    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_roc, train_target, train_pred = eval(model, device, loader)
        else:
            train_roc = 0

        train_roc_list.append(train_roc)
        print('train: {:.6f}\n'.format(train_roc))

    if args.output_model_dir is not '':
        print('saving to {}'.format(args.output_model_dir + '_model.pth'))
        torch.save(molecule_model.state_dict(), args.output_model_dir + '_model.pth')
        saved_model_dict = {
            'molecule_model': molecule_model.state_dict(),
            'model': model.state_dict(),
        }
        torch.save(saved_model_dict, args.output_model_dir + '_model_complete.pth')
