from typing import Tuple


import argparse
import json
import math
import os


import numpy
import torch
import torchvision


class Vae(torch.nn.Module):
    """Variational Autoencoder

    Args:
     input_d - input dimensions (height x width)
     n_frames - number of input flows (channels)
     n_latent - number of latent dimensions
     beta - KL loss multiplier for training
     batch - training batch size
     interpolation - interpolation mode for resize
     """

    def __init__(self,
                 input_d: Tuple[int],
                 n_frames: int,
                 n_latent: int,
                 beta: int,
                 batch: int,
                 interpolation: int = 0):
        super(Vae, self).__init__()
        self.n_frames = n_frames
        self.n_latent = n_latent
        self.input_d = input_d
        self.batch = batch
        self.beta = beta
        self.interpolation = interpolation

        self.layer_dims = []
        x, y = input_d
        for i in range(4):
            x = math.floor((x - 1) / 3 + 1)
            y = math.floor((y - 1) / 3 + 1)
            self.layer_dims.append((x, y))
        self.hidden_x = x
        self.hidden_y = y

        self.enc_conv1 = torch.nn.Conv2d(self.n_frames, 32, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn1 = torch.nn.BatchNorm2d(32)
        self.enc_af1 = torch.nn.ELU()
        
        self.enc_conv2 = torch.nn.Conv2d(32, 64, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn2 = torch.nn.BatchNorm2d(64)
        self.enc_af2 = torch.nn.ELU()

        self.enc_conv3 = torch.nn.Conv2d(64, 128, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn3 = torch.nn.BatchNorm2d(128)
        self.enc_af3 = torch.nn.ELU()

        self.enc_conv4 = torch.nn.Conv2d(128, 256, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn4 = torch.nn.BatchNorm2d(256)
        self.enc_af4 = torch.nn.ELU()

        self.linear_mu = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)
        self.linear_var = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)

        self.dec_linear = torch.nn.Linear(self.n_latent, 256 * self.hidden_x * self.hidden_y)

        self.dec_conv4 = torch.nn.ConvTranspose2d(256, 128, (5, 5),
            stride=(3, 3), padding=(2, 2),
            output_padding=(
                max(self.layer_dims[2][0] - 3 * (self.hidden_x - 1) - 1, 0),
                max(self.layer_dims[2][1] - 3 * (self.hidden_y - 1) - 1, 0)))
        self.dec_bn4 = torch.nn.BatchNorm2d(128)
        self.dec_af4 = torch.nn.ELU()
        print(self.layer_dims[2][0] - 3 * (self.hidden_x - 1) - 3)

        self.dec_conv3 = torch.nn.ConvTranspose2d(128, 64, (5, 5),
            stride=(3, 3), padding=(2, 2),
            output_padding=(
                max(self.layer_dims[1][0] - 3 * (self.layer_dims[2][0] - 1) - 1, 0),
                max(self.layer_dims[1][1] - 3 * (self.layer_dims[2][1] - 1) - 1, 0)))
        self.dec_bn3 = torch.nn.BatchNorm2d(64)
        self.dec_af3 = torch.nn.ELU()

        self.dec_conv2 = torch.nn.ConvTranspose2d(64, 32, (5, 5),
            stride=(3, 3), padding=(2, 2),
            output_padding=(
                max(self.layer_dims[0][0] - 3 * (self.layer_dims[1][0] - 1) - 1, 0),
                max(self.layer_dims[0][1] - 3 * (self.layer_dims[1][1] - 1) - 1, 0)))
        self.dec_bn2 = torch.nn.BatchNorm2d(32)
        self.dec_af2 = torch.nn.ELU()

        self.dec_conv1 = torch.nn.ConvTranspose2d(32, self.n_frames, (5, 5),
            stride=(3, 3), padding=(2, 2),
            output_padding=(
                max(self.input_d[0] - 3 * (self.layer_dims[0][0] - 1) - 1, 0),
                max(self.input_d[1] - 3 * (self.layer_dims[0][1] - 1) - 1, 0)))
        self.dec_bn1 = torch.nn.BatchNorm2d(self.n_frames)
        self.dec_af1 = torch.nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode an image as a distribution in latent space.

        Args:
            x - input image

        Returns:
            Tuple with 2 elements:
                0th: (n_latent x 1) tensor of means
                1st: (n_latent x 1) tensor of log variances
        """
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_af1(x)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_af2(x)
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.enc_af3(x)
        x = self.enc_conv4(x)
        x = self.enc_bn4(x)
        x = self.enc_af4(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)
        return mu, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a point in latent space to image space.
        
        Args:
            x - (n_latent x 1) sample from latent space

        Returns:
            Reconstructed image
        """
        z = self.dec_linear(z)
        z = torch.reshape(z, [self.batch, 256, self.hidden_x, self.hidden_y])
        z = self.dec_conv4(z)
        z = self.dec_bn4(z)
        z = self.dec_af4(z)
        z = self.dec_conv3(z)
        z = self.dec_bn3(z)
        z = self.dec_af3(z)
        z = self.dec_conv2(z)
        z = self.dec_bn2(z)
        z = self.dec_af2(z)
        z = self.dec_conv1(z)
        z = self.dec_bn1(z)
        return self.dec_af1(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode an image, sample its distribution, and decode back to image
        space.

        Args:
            x - input image

        Returns:
            Three element tuple:
                0th: reconstructe image
                1st: (n_latent x 1) tensor of means
                2nd: (n_latent x 1) tensor of log variances
        """
        mu, logvar = self.encode(x)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z)
        return out, mu, logvar

    def train_self(self,
                   train_path: str,
                   weights: str,
                   epochs: int,
                   use_flows: bool = False) -> None:
        """Train the VAE.

        Args:
            train_path - path to the training dataset
            weights - filename to save weights after training
            epochs - number of epochs to train for
            use_flows - if True, train a VAE on optical flows instead of raw 
                image data
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        model = self.to(device)
        
        def npy_loader(path: str) -> torch.Tensor:
            sample = torch.from_numpy(numpy.load(path))
            sample = torch.swapaxes(sample, 1, 2)
            sample = torch.swapaxes(sample, 0, 1)
            sample = sample.nan_to_num(0)
            sample = ((sample + 64) / 128).clamp(0, 1)
            sample = sample[0:self.n_frames, :, :]
            return sample.type(torch.FloatTensor) 

        if use_flows:
            train_set = torchvision.datasets.DatasetFolder(
                root=train_path,
                loader=npy_loader,
                extensions=['.npy'])
        elif self.n_frames == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d, interpolation=self.interpolation),
                torchvision.transforms.Grayscale()])
            train_set = torchvision.datasets.ImageFolder(
                root=train_path,
                transform=transforms)
        else:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d, interpolation=self.interpolation)])
            train_set = torchvision.datasets.ImageFolder(
                root=train_path,
                transform=transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            print('----------------------------------------------------')
            print(f'Epoch: {epoch}')

            model.train()
            epoch_tl = 0
            train_count = 0
            for data in train_loader:
                x, _ = data
                x = x.to(device)
                x_hat, mu, logvar = model(x)

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
                loss = mse_loss + self.beta * kl_loss
                epoch_tl += loss
                train_count += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Training Loss: {epoch_tl/train_count}')
            print('----------------------------------------------------')
        print('Training finished, saving weights...')
        torch.save(model, weights)


class VaeEncoder(torch.nn.Module):
    """Encoder portion only of VAE.

    Args:
        input_d - input dimensions (height x width)
        n_frames - input channels
        n_latent - number of latent dimensions
    """

    def __init__(self,
                 input_d: Tuple[int],
                 n_frames: int,
                 n_latent: int):
        super(VaeEncoder, self).__init__()
        self.n_frames = n_frames
        self.n_latent = n_latent
        self.input_d = input_d

        self.layer_dims = []
        x, y = input_d
        for i in range(4):
            x = math.floor((x - 1) / 3 + 1)
            y = math.floor((y - 1) / 3 + 1)
            self.layer_dims.append((x, y))
        self.hidden_x = x
        self.hidden_y = y

        self.enc_conv1 = torch.nn.Conv2d(self.n_frames, 32, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn1 = torch.nn.BatchNorm2d(32)
        self.enc_af1 = torch.nn.ELU()
        
        self.enc_conv2 = torch.nn.Conv2d(32, 64, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn2 = torch.nn.BatchNorm2d(64)
        self.enc_af2 = torch.nn.ELU()

        self.enc_conv3 = torch.nn.Conv2d(64, 128, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn3 = torch.nn.BatchNorm2d(128)
        self.enc_af3 = torch.nn.ELU()

        self.enc_conv4 = torch.nn.Conv2d(128, 256, (5, 5), stride=(3, 3), padding=(2, 2))
        self.enc_bn4 = torch.nn.BatchNorm2d(256)
        self.enc_af4 = torch.nn.ELU()

        self.linear_mu = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)
        self.linear_var = torch.nn.Linear(256 * self.hidden_x * self.hidden_y, self.n_latent)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode an image to latent space.

        Args:
            x - input image

        Returns:
            Two element tuple:
                0th: tensor of means
                1st: tensor of log variances
        """
        x = self.enc_conv1(x)
        x = self.enc_bn1(x)
        x = self.enc_af1(x)
        x = self.enc_conv2(x)
        x = self.enc_bn2(x)
        x = self.enc_af2(x)
        x = self.enc_conv3(x)
        x = self.enc_bn3(x)
        x = self.enc_af3(x)
        x = self.enc_conv4(x)
        x = self.enc_bn4(x)
        x = self.enc_af4(x)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)
        var = self.linear_var(x)
        return mu, var


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert VAE weights to Encoder weights only')
    parser.add_argument(
        '--weights',
        help='Path to weights file')
    args = parser.parse_args()
    original = torch.load(args.weights)
    new = VaeEncoder(
        input_d=original.input_d,
        n_frames=original.n_frames,
        n_latent=original.n_latent)
    new_state_dict = new.state_dict()
    for key in new.state_dict():
        new_state_dict[key] = original.state_dict()[key]
    new.load_state_dict(new_state_dict)
    new.eval()
    torch.save(new, f'{args.weights}_enc.pt')
