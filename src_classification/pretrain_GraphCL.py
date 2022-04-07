import argparse
import time

import numpy as np
import torch
import torch.optim as optim
from models import GNN
from pretrain_JOAO import graphcl
from torch_geometric.data import DataLoader

from datasets import MoleculeDataset_graphcl


def train(loader, model, optimizer, device):

    model.train()
    train_loss_accum = 0

    for step, (_, batch1, batch2) in enumerate(loader):
        # _, batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        x1 = model.forward_cl(batch1.x, batch1.edge_index,
                              batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index,
                              batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum / (step + 1)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='GraphCL')
    parser.add_argument('--device', type=int, default=0, help='gpu')
    parser.add_argument('--batch_size', type=int, default=256, help='batch')
    parser.add_argument('--decay', type=float, default=0, help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--JK', type=str, default="last",
                        choices=['last', 'sum', 'max', 'concat'],
                        help='how the node features across layers are combined.')
    parser.add_argument('--gnn_type', type=str, default="gin", help='gnn model type')
    parser.add_argument('--dropout_ratio', type=float, default=0, help='dropout ratio')
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions')
    parser.add_argument('--dataset', type=str, default=None, help='root dir of dataset')
    parser.add_argument('--num_layer', type=int, default=5, help='message passing layers')
    # parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset")
    parser.add_argument('--output_model_file', type=str, default='', help='model save path')
    parser.add_argument('--num_workers', type=int, default=8, help='workers for dataset loading')

    parser.add_argument('--aug_mode', type=str, default='sample')
    parser.add_argument('--aug_strength', type=float, default=0.2)

    # parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--output_model_dir', type=str, default='')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) \
        if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # set up dataset
    if 'GEOM' in args.dataset:
        dataset = MoleculeDataset_graphcl('../datasets/{}/'.format(args.dataset),
                                          dataset=args.dataset)
    dataset.set_augMode(args.aug_mode)
    dataset.set_augStrength(args.aug_strength)
    print(dataset)

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # set up model
    gnn = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK,
              drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = graphcl(gnn)
    model.to(device)

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    aug_prob = np.ones(25) / 25
    dataset.set_augProb(aug_prob)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        pretrain_loss = train(loader, model, optimizer, device)

        print('Epoch: {:3d}\tLoss:{:.3f}\tTime: {:.3f}:'.format(
            epoch, pretrain_loss, time.time() - start_time))

    if not args.output_model_dir == '':
        saver_dict = {'model': model.state_dict()}
        torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')
        torch.save(model.gnn.state_dict(), args.output_model_dir + '_model.pth')
