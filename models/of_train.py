#!/usr/bin/env python3
import argparse
from pickle import FALSE
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from of_network import Bi3DOF, Encoder, Decoder
from datasets import *
from utils import progress_bar

'''

Train routine for bi3dof and bi3dofopt methods in the paper "Improving Variational Autoencoder 
based Out-of-Distribution Detection for Embedded Real-time Applications" 

A bi3dof detector is trained by the default parameter values. 

To train a bi3dof-optprior detector, first compute the mu and var of optic flow fields of the
training data set. 

Then run the python script with arguments --latentprior optimal and mu&var values as described
in help of init_param().

More technical details please see in paper link: https://arxiv.org/abs/2107.11750

The bi3dof detector has been used for the case study in the paper "Design Methodology for Deep 
Out-of-Distribution Detectors in Real-Time Cyber-Physical Systems"

'''


def init_param():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_size", type=int, default=5,
                        help="number of videos in one mini-batch")
    parser.add_argument("--n_seq", type=int, default=6,
                        help="number of sequence to sample from one video")
    parser.add_argument("--n_frames", type=int, default=6,
                        help="number of frames in a 3D input cube")
    parser.add_argument("--group", type=int, default=2,
                        help="2 refer to the two latent subspaces, in horizontal and vertical direction respectively.")
    parser.add_argument("--n_latents", type=int, default=12,
                        help="Dimension of one latent subspace")
    parser.add_argument("--latentprior", type=str, default='simple',
                        help="If 'optimal' the following mu and var values will be used to compute distribution descrepancy in latent space.")
    parser.add_argument("--mu1", type=float, default=0.,
                        help="Mean optic flow fields in the horizontal direction. Need to be computed standalone from a specific trainig set.")
    parser.add_argument("--mu2", type=float, default=0.,
                        help="Mean optic flow value in the vertical direction")
    parser.add_argument("--var1", type=float, default=1.,
                        help="Variance in the horizontal direction")
    parser.add_argument("--var2", type=float, default=1.,
                        help="Variance in the vertical direction")
    parser.add_argument("--img_height", type=int, default=120,
                        help="Height of the input images, e.g. 120 / 90 / 60")
    parser.add_argument("--img_width", type=int, default=160,
                        help="Width of the input images, e.g. 160 / 120 / 80")
    parser.add_argument("--crop_height", type=int, default=113,
                        help="Height of the input images, e.g. 113 / 87 / 56")
    parser.add_argument("--crop_width", type=int, default=152,
                        help="Width of the input images, e.g. 152 / 116 / 77")
    args = parser.parse_args()

    # Manual input dimension as in # [g,d,h,w]
    # args.input_size = [args.group, args.n_frames, 120, 160]
    # args.transform_size = [113, 152]
    # args.input_size = [args.group, args.n_frames, 90, 120]
    # args.transform_size = [87, 116]
    # args.input_size = [args.group, args.n_frames, 60, 80]
    # args.transform_size = [56, 77]

    # Multi-size training
    args.input_size = [args.group, args.n_frames,
                       args.img_height, args.img_width]
    args.transform_size = [args.crop_height, args.crop_width]
    print(
        f"Image Size: {args.img_height}x{args.img_width} Crop Size: {args.crop_height}x{args.crop_width}")

    # default values ep 600, LR 0.0001
    args.epochs = 600
    args.lr_base = 0.0001
    args.kl_weight = 1
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Running on %s" % args.device)

    if args.dummy:
        args.epochs = 1

    return args


def run():
    # load data
    train_id = Bi3DOFDataset(args)
    train_size = len(train_id)
    train_loader = torch.utils.data.DataLoader(
        train_id, batch_size=args.episode_size, shuffle=True, num_workers=1)

    # initialize network and optimizor
    encoder = Encoder(args)
    decoder = Decoder(args)
    vae = Bi3DOF(encoder, decoder, args).to(args.device)
    enc_optimizer = optim.Adam(vae.encoder.parameters(), lr=args.lr_base)
    dec_optimizer = optim.Adam(vae.decoder.parameters(), lr=args.lr_base)

    for epoch in range(args.epochs):
        loss_reconstruction, loss_latent, num_examples = 0, 0, 0
        vae.train()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = batch_data.to(args.device)
            (b1, b2, g, d, h, w) = batch_data.shape
            batch_data = batch_data.view((b1*b2, g, d, h, w))
            num_examples += b1*b2

            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss, loss_rec, (loss_latent_grp1, loss_latent_grp2) = vae.loss(
                batch_data, args.kl_weight)
            loss = loss.mean(dim=-1)
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()

            loss_reconstruction += loss_rec.view(-1).mean()
            loss_latent += (loss_latent_grp1.view(-1).mean() +
                            loss_latent_grp2.view(-1).mean())

            progress_bar(batch_idx, len(train_loader), 'Epoch%3d  Recontruction/Loss: %.6f  Latent/Loss: %.6f'
                         % (epoch, loss_reconstruction/num_examples, loss_latent/num_examples))
    # save
    # torch.save(vae.state_dict(),
    #            "bi3dof-duckie-{}-{}epoch.pt".format(args.latentprior,  epoch+1))
    torch.save(vae.state_dict(),
               "bi3dof_{}x{}_{}x{}.pt".format(args.img_height, args.img_width, args.crop_height, args.crop_width))


#
seed()
args = init_param()
args.training = True
# Need to change the data file that is created through feature_abstraction
# args.data_file = "datasets/duckietown_orig.train"
args.data_file = "datasets/" + \
    str(args.img_height) + "x" + str(args.img_width) + "/duckietown.train"
print(f"Dataset from {args.data_file}")
run()
