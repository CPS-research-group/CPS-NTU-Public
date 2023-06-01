#!/usr/bin/env python3
import argparse
import torch
import json
from icp import Icp
from vae import Vae


class OodDetector(torch.nn.Module):

    def __init__(self, vae_weights, icp_weights):
        super().__init__()
        self.vae = torch.load(vae_weights)
        with open(icp_weights, 'r') as f:
            cal_data = json.loads(f.read())
            self.mask = torch.zeros(self.vae.n_latent)
            self.mask[cal_data['rain']['top_z'][0][0]] = 1
            cal_set = torch.sum(self.mask * torch.Tensor(cal_data['rain']['dkls']), dim=1)
            self.icp = Icp(cal_set)

    def forward(self, x):
        mu, logvar, _, _ = self.vae.encoder(x)
        kl_loss = 0.5 * torch.sum(self.mask * (mu.pow(2) + logvar.exp() - logvar - 1))
        return 1 - self.icp(kl_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Form an OOD detector from various weights')
    parser.add_argument(
        '--vae_weights',
        help='*.pt file with the VAE weights'
    )
    parser.add_argument(
        '--icp_weights',
        help='*.json file with the ICP weights'
    )
    parser.add_argument(
        '--name',
        help='name for the new ood detector'
    )
    args = parser.parse_args()
    detector = OodDetector(
        vae_weights=args.vae_weights,
        icp_weights=args.icp_weights
    )
    torch.save(detector, args.name)

