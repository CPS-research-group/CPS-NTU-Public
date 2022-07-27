#!/usr/bin/env python3
"""Find the number of latent dimensions and the beta value that maximize the
mutual information gain for a Beta VAE."""

from typing import Tuple
import argparse
import math
import sys
import numpy
import torch
import torchvision


from scipy.stats import gaussian_kde
from bayes_opt import BayesianOptimization
from bvae import BetaVae

N_CHAN = 3
INPUT_DIM = (224, 224)
BATCH = 64
TRAIN_PATH = ''
CV_PATH = ''


def mig(n, beta) -> float:
    """Find the mutual information gain for an instance of the beta
    variational autoencoder network.

    Args:
        n: number of latent dimensions
        beta: weight of KL-divergence loss during training
    """
    n = int(n)
    model = BetaVae(
        n_latent=n,
        beta=beta,
        n_chan=N_CHAN,
        input_d=INPUT_DIM,
        batch=BATCH)
    model.train_self(
        data_path=TRAIN_PATH,
        epochs=100,
        weights_file=f'bvae_n{n}_b{beta}_{"bw" if N_CHAN == 1 else ""}_'
                     f'{INPUT_DIM[0]}x{INPUT_DIM[1]}.pt')
    return model.mig(CV_PATH, iters=5)


def optimize_mig():
    """Find the parameter n_latent  and beta that maximize the mutual
    information gain of a beta variational autoencoder"""
    optimizer = BayesianOptimization(
        f=mig,
        pbounds={'n': (30, 200), 'beta': (1,5)},
        verbose=2,
        random_state=1)
    optimizer.maximize(n_iter=50)
    print('#################################################################')
    print(f'Found Network with Optimal MIG of {optimizer.max["target"]}')
    print(f'Parameters: {optimizer.max["params"]}')
    print('#################################################################')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Tune hyperparameters for BVAE model.')
    parser.add_argument(
        '--training_set',
        help='Path to training set')
    parser.add_argument(
        '--cv_set',
        help='Path to cross-validation set.')
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Use grayscale images')
    parser.add_argument(
        '--dimensions',
        help='Dimensions of input image')
    parser.add_argument(
        '--batch',
        help='Batch size to use for training.')
    args = parser.parse_args()
    N_CHAN = 1 if args.grayscale else 3
    INPUT_DIM = tuple([int(i) for i in args.dimensions.split('x')])
    BATCH = int(args.batch)
    TRAIN_PATH = args.training_set
    CV_PATH = args.cv_set
    optimize_mig()
