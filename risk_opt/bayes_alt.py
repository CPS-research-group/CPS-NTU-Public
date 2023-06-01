#!/usr/bin/env python3
"""Find the number of latent dimensions and the beta value that maximize the
mutual information gain for a Beta VAE."""

from typing import Tuple
import argparse
import math
import sys
import numpy as np
import torch
import torchvision


from scipy.stats import gaussian_kde
from bayes_opt import BayesianOptimization




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

RISK = np.load('risk_0.2.npy')
print(RISK.shape)
def target(yolo_t, yolo_s, ood_t, ood_s):
    out = -RISK[int(ood_t), int(ood_s), int(yolo_t), int(yolo_s)]
    return out


"""Find the parameter n_latent  and beta that maximize the mutual
information gain of a beta variational autoencoder"""
optimizer = BayesianOptimization(
    f=target,
    pbounds={'yolo_t': (0, 99), 'yolo_s': (0,13), 'ood_t': (0, 99), 'ood_s': (0,27)},
    verbose=2,
    random_state=1)
optimizer.maximize(n_iter=1000)
print('#################################################################')
print(f'Found Network with Optimal MIG of {optimizer.max["target"]}')
print(f'Parameters: {optimizer.max["params"]}')
print('#################################################################')
print(np.amin(RISK))