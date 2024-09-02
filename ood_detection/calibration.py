#!/usr/bin/env python3
"""Find the latent variables that correspond to each generative factor in the
BetaVAE OOD detector.  A file of KL divergence scores for all the samples in
the calibration set will be returned (needed for BetaVAE test-time detection),
as well as the rank of each latent variable with respect to its effect on each
generative factor."""


from typing import Dict, List, Tuple
import argparse
import json
import os
import cv2
import numpy
import torch
import torchvision
from quantization.quant_bvae import BetaVae
import sys
numpy.set_printoptions(threshold=sys.maxsize)


def get_scene_dkls(scene: str, network: torch.nn.Module) -> List[List[float]]:
    """Get the KL divergence for each frame in a given scene.
    
    Args:
        scene - path to scene (video file) to process.
        network - torch model whose output is a tuple where (out, mu, logvar)
            represents: *out* - decoder output, *mu* - sample mean in latent
            space, and *logvar* - sample log variance in latent space.
    
    Returns:
        A list of (1 x n_latent) lists corresponding to the KL divergence for
        each frame in the scene for each latent variable in the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dkls = []
    vid_in = cv2.VideoCapture(scene)
    while vid_in.isOpened():
        ret, frame = vid_in.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, network.input_d)
        if network.n_chan == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = torchvision.transforms.functional.to_tensor(frame)
        _, mu, logvar = network(frame.unsqueeze(0).to(device))
        mu = mu.detach().cpu().numpy()
        logvar = logvar.detach().cpu().numpy()
        dkl = 0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)
        dkls.append(numpy.squeeze(dkl).tolist())
    return dkls


def get_scene_kl_diff(dkls: List[List[float]]) -> numpy.ndarray:
    """Get the mean KLdiff for a given scene.
    
    Args:
        dkls - A list of KL divergeneces for each frame in a scene.

    Returns:
        A (1 x n_latent) array of mean KLdiff for each latent dimension for
        the provided scene and model.
    """
    kl_curr = None
    kl_next = None
    scene_length = 0
    scene_mean = numpy.zeros((1, len(dkls[0])))
    for frame in dkls:
        kl_curr = kl_next
        kl_next = numpy.array(frame).reshape((1, len(frame)))
        if kl_curr is None:
            continue
        else:
            kl_diff = numpy.abs(kl_next - kl_curr)
            scene_length +=1
            scene_mean += (kl_diff - scene_mean) / scene_length
    return scene_mean


def  get_partition_variance(partition_path: str,
                           network: torch.nn.Module) -> Dict[str,
                                                             List[Tuple]]:
    """Get the variance of KLdiff for a partition.
    
    Args:
        partition_path - path to partition in calibration set.
        network - torch model whose output is a tuple where (out, mu, logvar)
            represents: *out* - decoder output, *mu* - sample mean in latent
            space, and *logvar* - sample log variance in latent space.
    
    Returns:
        Dictionary with keys:
            dkls - list of KL divergences for each latent dimension in each
                frame in the partition.
            top_z - sorted list of tuples of the form (latent dimension, 
                variance), where the first tuple is the latent dimension with
                the highest variance.
    """
    partition_stats = {'dkls': [], 'top_z': []}
    wellford_m2 = numpy.zeros((1, network.n_latent))
    wellford_mean = numpy.zeros((1, network.n_latent))
    wellford_count = 0
    i = 0
    print(len(os.listdir(partition_path)))
    for scene in os.listdir(partition_path):
        print(i) 
        i+=1

        dkls = get_scene_dkls(os.path.join(partition_path, scene), network)
        partition_stats['dkls'].extend(dkls)
        scene_mean = get_scene_kl_diff(dkls)
        wellford_count += 1
        delta = scene_mean - wellford_mean
        wellford_m2 = numpy.power(delta, 2)
        variance = wellford_m2 / wellford_count

    variance = numpy.squeeze(variance).tolist()
    for _ in range(len(variance)):
        idx, val = max(enumerate(variance), key=lambda x: x[1])
        partition_stats['top_z'].append((idx, val))
        variance[idx] = -1
    return partition_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Find most informative latent variables '
                                     'for each generative factor as well as '
                                     'the KL divergence scores for each '
                                     'sample in the calibaration set.')
    parser.add_argument(
        '--weights',
        help='Path to weights file for the network being calibrated.')
    parser.add_argument(
        '--dataset',
        help='Path to calibration dataset.  This should be a folder '
             'containing subfolders for each data partition.  Each partition '
             'folder should contain videos for each generative factor level '
             'present in the training set.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    network = torch.load(args.weights)
    network.batch = 1
    network.eval()
    network.to(device)
    alpha_cal = {}
    alpha_cal['PARAMS'] = {}
    alpha_cal['PARAMS']['n_latent'] = network.n_latent
    alpha_cal['PARAMS']['input_d'] = network.input_d
    alpha_cal['PARAMS']['n_chan'] = network.n_chan

    for partition in os.listdir(args.dataset):
        print(f'Processing partition: {partition}')
        alpha_cal[partition] = get_partition_variance(
            os.path.join(args.dataset, partition),
            network)
        print(f'Rankings for partition: {partition}')
        for rank, value in enumerate(alpha_cal[partition]['top_z']):
            print(f'{rank}: {value}')

    dest_path = list(os.path.split(args.weights))
    dest_path[-1] = f'alpha_cal_{dest_path[-1].replace("pt", "json")}'        
    with open(os.path.join(*dest_path), 'w') as alpha_cal_f:
        alpha_cal_f.write(json.dumps(alpha_cal))
