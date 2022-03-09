# import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import args
from dataloader import (DataLoaderAE, DataLoaderMasking,
                        DataLoaderSubstructContext3D)
from models import (GNN, AutoEncoder, Discriminator,
                    EnergyVariationalAutoEncoder, SchNet,
                    VariationalAutoEncoder)
from pretrain_AM import do_AttrMasking
from pretrain_CP import do_ContextPred
from pretrain_EP import do_EdgePred
from pretrain_IG import do_InfoGraph
# from itertools import repeat
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
from util import (ExtractSubstructureContextPair, MaskAtom, NegativeEdge,
                  cycle, do_GraphCL, do_GraphCLv2, dual_CL,
                  update_augmentation_probability_JOAO,
                  update_augmentation_probability_JOAOv2)

from datasets import Molecule3DDataset, MoleculeDataset_graphcl


def save_model(save_best, epoch=None):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model.pth')
            saver_dict = {
                'model':    molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete.pth')

        elif epoch is None:
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model_final.pth')
            saver_dict = {
                'model':    molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_final.pth')

        else:
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + '_model_{}.pth'.format(epoch))
            saver_dict = {
                'model':    molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + '_model_complete_{}.pth'.format(epoch))

    return


def train_no_aug(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    for support_model in SSL_2D_support_model_list:
        support_model.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    SSL_2D_loss_accum, SSL_2D_acc_accum = 0, 0

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for batch in l:
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)
        molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch)

        ##### To obtain 3D-2D SSL loss and acc
        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        if args.AE_model == 'Energy_VAE':
            AE_loss_1, AE_acc_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2, AE_acc_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
        else:
            AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
            AE_acc_1 = AE_acc_2 = 0
        AE_loss = (AE_loss_1 + AE_loss_2) / 2

        ##### To obtain 2D SSL loss and acc
        if args.SSL_2D_mode == 'EP':
            SSL_2D_loss, SSL_2D_acc = do_EdgePred(
                node_repr=node_repr, batch=batch, criterion=criterion)

        elif args.SSL_2D_mode == 'IG':
            SSL_2D_loss, SSL_2D_acc = do_InfoGraph(
                node_repr=node_repr, batch=batch,
                molecule_repr=molecule_2D_repr, criterion=criterion,
                infograph_discriminator_SSL_model=infograph_discriminator_SSL_model)

        elif args.SSL_2D_mode == 'AM':
            masked_node_repr = molecule_model_2D(batch.masked_x, batch.edge_index, batch.edge_attr)
            SSL_2D_loss, SSL_2D_acc = do_AttrMasking(
                batch=batch, criterion=criterion, node_repr=masked_node_repr,
                molecule_atom_masking_model=molecule_atom_masking_model)

        elif args.SSL_2D_mode == 'CP':
            SSL_2D_loss, SSL_2D_acc = do_ContextPred(
                batch=batch, criterion=criterion, args=args,
                molecule_substruct_model=molecule_model_2D,
                molecule_context_model=molecule_context_model,
                molecule_readout_func=molecule_readout_func)

        else:
            raise Exception

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        AE_loss_accum += AE_loss.detach().cpu().item()
        AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2
        SSL_2D_loss_accum += SSL_2D_loss.detach().cpu().item()
        SSL_2D_acc_accum += SSL_2D_acc

        loss = 0
        if args.alpha_1 > 0:
            loss += CL_loss * args.alpha_1
        if args.alpha_2 > 0:
            loss += AE_loss * args.alpha_2
        if args.alpha_3 > 0:
            loss += SSL_2D_loss * args.alpha_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    SSL_2D_loss_accum /= len(loader)
    SSL_2D_acc_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum + args.alpha_3 * SSL_2D_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print(
        'CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tAE Acc: {:.5f}\t2D SSL Loss: {:.5f}\t2D SSL Acc: {:.5f}Time: {:.5f}'.format(
            CL_loss_accum, CL_acc_accum, AE_loss_accum, AE_acc_accum, SSL_2D_loss_accum, SSL_2D_acc_accum,
            time.time() - start_time))
    return


def train_with_aug(args, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()
    for support_model in SSL_2D_support_model_list:
        support_model.train()

    AE_loss_accum, AE_acc_accum = 0, 0
    CL_loss_accum, CL_acc_accum = 0, 0
    SSL_2D_loss_accum, SSL_2D_acc_accum = 0, 0

    aug_prob_cp = loader.dataset.aug_prob
    n_aug = np.random.choice(25, 1, p=aug_prob_cp)[0]
    n_aug1, n_aug2 = n_aug // 5, n_aug % 5

    if args.verbose:
        l = tqdm(loader)
    else:
        l = loader
    for batch, batch1, batch2 in l:
        batch = batch.to(device)
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)
        molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch)

        ##### To obtain 3D-2D SSL loss and acc
        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)
        if args.AE_model == 'Energy_VAE':
            AE_loss_1, AE_acc_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2, AE_acc_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
        else:
            AE_loss_1 = AE_2D_3D_model(molecule_2D_repr, molecule_3D_repr)
            AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_2D_repr)
            AE_acc_1 = AE_acc_2 = 0
        AE_loss = (AE_loss_1 + AE_loss_2) / 2

        ##### To obtain 2D SSL loss and acc
        if args.SSL_2D_mode in ['GraphCL', 'JOAO']:
            SSL_2D_loss = do_GraphCL(
                batch1=batch1, batch2=batch2,
                molecule_model_2D=molecule_model_2D, projection_head=projection_head,
                molecule_readout_func=molecule_readout_func)
        elif args.SSL_2D_mode == 'JOAOv2':
            SSL_2D_loss = do_GraphCLv2(
                batch1=batch1, batch2=batch2, n_aug1=n_aug1, n_aug2=n_aug2,
                molecule_model_2D=molecule_model_2D, projection_head=projection_head,
                molecule_readout_func=molecule_readout_func)
        SSL_2D_acc = 0

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        AE_loss_accum += AE_loss.detach().cpu().item()
        AE_acc_accum += (AE_acc_1 + AE_acc_2) / 2
        SSL_2D_loss_accum += SSL_2D_loss.detach().cpu().item()
        SSL_2D_acc_accum += SSL_2D_acc

        loss = 0
        if args.alpha_1 > 0:
            loss += CL_loss * args.alpha_1
        if args.alpha_2 > 0:
            loss += AE_loss * args.alpha_2
        if args.alpha_3 > 0:
            loss += SSL_2D_loss * args.alpha_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global aug_prob
    if args.SSL_2D_mode == 'JOAO':
        aug_prob = update_augmentation_probability_JOAO(
            loader=loader, molecule_model_2D=molecule_model_2D, projection_head=projection_head,
            molecule_readout_func=molecule_readout_func,
            gamma_joao=args.gamma_joao, device=device)
    elif args.SSL_2D_mode == 'JOAOv2':
        aug_prob = update_augmentation_probability_JOAOv2(
            loader=loader, molecule_model_2D=molecule_model_2D, projection_head=projection_head,
            molecule_readout_func=molecule_readout_func,
            gamma_joao=args.gamma_joaov2, device=device)

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    AE_acc_accum /= len(loader)
    SSL_2D_loss_accum /= len(loader)
    SSL_2D_acc_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum + args.alpha_3 * SSL_2D_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print(
        'CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tAE Acc: {:.5f}\t2D SSL Loss: {:.5f}\t2D SSL Acc: {:.5f}Time: {:.5f}'.format(
            CL_loss_accum, CL_acc_accum, AE_loss_accum, AE_acc_accum, SSL_2D_loss_accum, SSL_2D_acc_accum,
            time.time() - start_time))
    return


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda:' + str(args.device)) \
        if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.cuda.set_device(args.device)

    print('========== Use {} for 2D SSL =========='.format(args.SSL_2D_mode))

    transform = None
    criterion = None
    if args.SSL_2D_mode == 'EP':
        transform = NegativeEdge()
        criterion = nn.BCEWithLogitsLoss()
    elif args.SSL_2D_mode == 'AM':
        transform = MaskAtom(num_atom_type=119, num_edge_type=5,
                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)
        criterion = nn.CrossEntropyLoss()
    elif args.SSL_2D_mode == 'CP':
        l1 = args.num_layer - 1
        l2 = l1 + args.csize
        transform = ExtractSubstructureContextPair(args.num_layer, l1, l2)
        criterion = nn.BCEWithLogitsLoss()
    elif args.SSL_2D_mode == 'IG':
        criterion = nn.BCEWithLogitsLoss()

    data_root = '../datasets/{}/'.format(args.dataset) \
        if args.input_data_dir == '' \
        else '{}/{}/'.format(args.input_data_dir, args.dataset)
    # assert args.dataset in [
    #     'GEOM_3D_02', 'GEOM_3D_03', 'GEOM_3D_04', 'GEOM_01_3D', 'GEOM_test',
    #     'GEOM_01_3D_New', 'GEOM_02_3D_New', 'GEOM_3D_nmol1000000_nconf1', 'GEOM_3D_nmol1000000_nconf5']

    if args.SSL_2D_mode in ['GraphCL', 'JOAO', 'JOAOv2']:
        dataset = MoleculeDataset_graphcl(data_root, dataset=args.dataset)
        dataset.set_augMode('sample')
        dataset.set_augStrength(0.2)
    else:
        dataset = Molecule3DDataset(data_root, dataset=args.dataset, transform=transform)

    if args.SSL_2D_mode == 'EP':
        loader = DataLoaderAE(dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    elif args.SSL_2D_mode == 'AM':
        loader = DataLoaderMasking(dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
    elif args.SSL_2D_mode == 'CP':
        loader = DataLoaderSubstructContext3D(dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=args.num_workers)
    else:  # IG, GraphCL, JOAO, JOAOv2
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    # set up 2D and 3D base model
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK,
                            drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
    molecule_readout_func = global_mean_pool
    molecule_model_3D = SchNet(
        hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

    # set up VAE model
    if args.AE_model == 'Energy_VAE':
        AE_2D_3D_model = EnergyVariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
            beta=args.beta, args=args).to(device)
        AE_3D_2D_model = EnergyVariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
            beta=args.beta, args=args).to(device)
    elif args.AE_model == 'AE':
        AE_2D_3D_model = AutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
        AE_3D_2D_model = AutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target).to(device)
    elif args.AE_model == 'VAE':
        AE_2D_3D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
            beta=args.beta).to(device)
        AE_3D_2D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target,
            beta=args.beta).to(device)
    else:
        raise Exception

    # set up 2D SSL model
    SSL_2D_support_model_list = []

    # set up 2D SSL model for IG
    infograph_discriminator_SSL_model = None
    if args.SSL_2D_mode == 'IG':
        infograph_discriminator_SSL_model = Discriminator(args.emb_dim).to(device)
        SSL_2D_support_model_list.append(infograph_discriminator_SSL_model)

    # set up 2D SSL model for AM
    molecule_atom_masking_model = None
    if args.SSL_2D_mode == 'AM':
        molecule_atom_masking_model = torch.nn.Linear(args.emb_dim, 119).to(device)
        SSL_2D_support_model_list.append(molecule_atom_masking_model)

    # set up 2D SSL model for CP
    molecule_context_model = None
    if args.SSL_2D_mode == 'CP':
        molecule_context_model = GNN(int(l2 - l1), args.emb_dim, JK=args.JK,
                                     drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type).to(device)
        SSL_2D_support_model_list.append(molecule_context_model)

    # set up 2D SSL model for GraphCL, JOAO, JOAOv2
    projection_head = None
    if args.SSL_2D_mode in ['GraphCL', 'JOAO']:
        projection_head = nn.Sequential(nn.Linear(300, 300),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(300, 300)).to(device)
        SSL_2D_support_model_list.append(projection_head)
    if args.SSL_2D_mode == 'JOAOv2':
        projection_head = nn.ModuleList([
            nn.Sequential(nn.Linear(300, 300),
                          nn.ReLU(inplace=True),
                          nn.Linear(300, 300))
            for _ in range(5)]).to(device)
        SSL_2D_support_model_list.append(projection_head)

    # set up parameters
    model_param_group = [{'params': molecule_model_2D.parameters(),
                          'lr':     args.lr * args.gnn_lr_scale},
                         {'params': molecule_model_3D.parameters(),
                          'lr':     args.lr * args.schnet_lr_scale},
                         {'params': AE_2D_3D_model.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},
                         {'params': AE_3D_2D_model.parameters(),
                          'lr': args.lr * args.schnet_lr_scale}]
    for SSL_2D_support_model in SSL_2D_support_model_list:
        model_param_group.append({'params': SSL_2D_support_model.parameters(),
                                  'lr': args.lr * args.gnn_lr_scale})

    # set up optimizers
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    # for GraphCL, JOAO, JOAOv2
    aug_prob = np.ones(25) / 25
    np.set_printoptions(precision=3, floatmode='fixed')

    if args.SSL_2D_mode in ['GraphCL', 'JOAO', 'JOAOv2']:
        train_function = train_with_aug
    else:
        train_function = train_no_aug

    # start training
    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))

        if args.SSL_2D_mode in ['JOAO', 'JOAOv2']:
            dataset.set_augProb(aug_prob)
            print('augmentation probability\t', aug_prob)

        train_function(args, device, loader, optimizer)

        if epoch == 50:
            save_model(save_best=False, epoch=50)

    if args.SSL_2D_mode in ['JOAO', 'JOAOv2']:
        print('augmentation probability\t', aug_prob)

    # save final model weight
    save_model(save_best=False)
