import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, '../src_classification')
from config import args
from models import AutoEncoder, SchNet, VariationalAutoEncoder
from models_complete_feature import GNNComplete
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from util import dual_CL

from datasets import Molecule3DMaskingDataset


def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        else:
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')
    return


def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    if molecule_projection_layer is not None:
        molecule_projection_layer.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if args.model_3d == 'schnet':
            molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch)

        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)

        AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
        AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
        AE_acc_1 = AE_acc_2 = 0
        AE_loss = (AE_loss_1 + AE_loss_2) / 2

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        AE_loss_accum += AE_loss.detach().cpu().item()
        AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2

        loss = 0
        if args.alpha_1 > 0:
            loss += CL_loss * args.alpha_1
        if args.alpha_2 > 0:
            loss += AE_loss * args.alpha_2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tAE Acc: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, AE_loss_accum, AE_acc_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    if 'GEOM' in args.dataset:
        data_root = '../datasets/{}/'.format(args.dataset) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset)
        dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio)
    else:
        raise Exception
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    molecule_model_2D = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool

    print('Using 3d model\t', args.model_3d)
    molecule_projection_layer = None
    if args.model_3d == 'schnet':
        molecule_model_3D = SchNet(
            hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    else:
        raise NotImplementedError('Model {} not included.'.format(args.model_3d))

    if args.AE_model == 'AE':
        AE_2D_3D_model = AutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
        AE_3D_2D_model = AutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
    elif args.AE_model == 'VAE':
        AE_2D_3D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
        AE_3D_2D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    else:
        raise Exception

    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    model_param_group.append({'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    if molecule_projection_layer is not None:
        model_param_group.append({'params': molecule_projection_layer.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, device, loader, optimizer)

    save_model(save_best=False)
