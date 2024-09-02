#!/usr/bin/env python3
"""BetaVAE OOD detector module.  This module contains everything needed to
train the BetaVAE OOD detector and convert the trained model to an encoder-
only model for efficient test-time execution."""

from torch.quantization import QuantStub, DeQuantStub

from typing import Tuple
import math
from pyparsing import col
import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader

class QuantBetaVae(torch.nn.Module):
    """BetaVAE OOD detector.  This class includes both the encoder and
    decoder portions of the model.
    
    Args:
        n_latent - number of latent dimensions
        beta - hyperparameter beta to use during training
        n_chan - number of channels in the input image
        input_d - height x width tuple of input image size in pixels
    """

    def __init__(self,
                 n_latent: int,
                 beta: float,
                 n_chan: int,
                 input_d: Tuple[int],
                 batch: int = 1) -> None:
        super(QuantBetaVae, self).__init__()
        self.batch = batch
        self.n_latent = n_latent
        self.beta = beta
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.enc_conv1 = torch.nn.Conv2d(
            in_channels=self.n_chan,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv1_bn = torch.nn.BatchNorm2d(128)
        self.enc_conv1_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv1_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv2_bn = torch.nn.BatchNorm2d(64)
        self.enc_conv2_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv2_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv3_bn = torch.nn.BatchNorm2d(32)
        self.enc_conv3_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv3_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv4_bn = torch.nn.BatchNorm2d(16)
        self.enc_conv4_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv4_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        self.enc_dense1_af = torch.nn.LeakyReLU(0.1)
        
        self.enc_dense2 = torch.nn.Linear(2048, 1000)
        self.enc_dense2_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4mu_af = torch.nn.LeakyReLU(0.1)
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense4 = torch.nn.Linear(self.n_latent, 250)
        self.dec_dense4_af = torch.nn.LeakyReLU(0.1)
        
        self.dec_dense3 = torch.nn.Linear(250, 1000)
        self.dec_dense3_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense2 = torch.nn.Linear(1000, 2048)
        self.dec_dense2_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense1 = torch.nn.Linear(2048, self.hidden_units)
        self.dec_dense1_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv4_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv4 = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv4_bn = torch.nn.BatchNorm2d(32)
        self.dec_conv4_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv3_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv3 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv3_bn = torch.nn.BatchNorm2d(64)
        self.dec_conv3_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv2_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv2 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv2_bn = torch.nn.BatchNorm2d(128)
        self.dec_conv2_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv1_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv1 = torch.nn.ConvTranspose2d(
            in_channels=128,
            out_channels=self.n_chan,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv1_bn = torch.nn.BatchNorm2d(self.n_chan)
        self.dec_conv1_af = torch.nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode tensor x to its latent representation.
        
        Args:
            x - batch x channels x height x width tensor.

        Returns:
            (mu, var) where mu is sample mean and var is log variance in
            latent space.
        """

        # z = x
        z = self.quant(x)

        z = self.enc_conv1(z)
        z = self.enc_conv1_bn(z)
        z = self.enc_conv1_af(z)

        z = self.dequant(z)
        z, self.indices1 = self.enc_conv1_pool(z)
        z = self.quant(z)

        z = self.enc_conv2(z)
        z = self.enc_conv2_bn(z)
        z = self.enc_conv2_af(z)

        z = self.dequant(z)
        z, self.indices2 = self.enc_conv2_pool(z)
        z = self.quant(z)

        z = self.enc_conv3(z)
        z = self.enc_conv3_bn(z)
        z = self.enc_conv3_af(z)

        z = self.dequant(z)
        z, self.indices3 = self.enc_conv3_pool(z)
        z = self.quant(z)

        z = self.enc_conv4(z)
        z = self.enc_conv4_bn(z)
        z = self.enc_conv4_af(z)

        z = self.dequant(z)
        z, self.indices4 = self.enc_conv4_pool(z)
        z = self.quant(z)

   
        z = z.reshape(z.size(0), -1)
        # z = z.view(z.size(0), -1)
        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        var = self.enc_dense4var(z)
        var = self.enc_dense4var_af(var)

        z = self.dequant(z)

        return mu, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation to generate a reconstructed image.

        Args:
            z - 1 x n_latent input tensor.

        Returns:
            A batch x channels x height x width tensor representing the
            reconstructed image.
        """
        z  = self.quant(z)
        y = self.dec_dense4(z)
        y = self.dec_dense4_af(y)

        y = self.dec_dense3(y)
        y = self.dec_dense3_af(y)

        y = self.dec_dense2(y)
        y = self.dec_dense2_af(y)

        y = self.dec_dense1(y)
        y = self.dec_dense1_af(y)

        y = torch.reshape(y, [self.batch, 16, self.y_5, self.x_5])

        y = self.dequant(y)
        y = self.dec_conv4_pool(
            y,
            self.indices4,
            output_size=torch.Size([self.batch, 16, self.y_4, self.x_4]))
        y = self.quant(y)
        
        y = self.dec_conv4(y)
        y = self.dec_conv4_bn(y)
        y = self.dec_conv4_af(y)

        y = self.dequant(y)
        y = self.dec_conv3_pool(
            y,
            self.indices3,
            output_size=torch.Size([self.batch, 32, self.y_3, self.x_3]))
        y = self.quant(y)

        y = self.dec_conv3(y)
        y = self.dec_conv3_bn(y)
        y = self.dec_conv3_af(y)

        y = self.dequant(y)
        y = self.dec_conv2_pool(
            y,
            self.indices2,
            output_size=torch.Size([self.batch, 64, self.y_2, self.x_2]))
        y = self.quant(y)

        y = self.dec_conv2(y)
        y = self.dec_conv2_bn(y)
        y = self.dec_conv2_af(y)

        y = self.dequant(y)
        y = self.dec_conv1_pool(
            y,
            self.indices1,
            output_size=torch.Size([self.batch, 128, self.input_d[0], self.input_d[1]]))
        y = self.quant(y)

        y = self.dec_conv1(y)
        y = self.dec_conv1_bn(y)
        y = self.dec_conv1_af(y)
        
        y = self.dequant(y)

        return y

    def get_layer_size(self, layer: int) -> Tuple[int]:
        """Given a network with some input size, calculate the dimensions of
        the resulting layers.
        
        Args:
            layer - layer number (for the encoder: 1 -> 2 -> 3 -> 4, for the
                decoder: 4 -> 3 -> 2 -> 1).

        Returns:
            (y, x) where y is the layer height in pixels and x is the layer
            width in pixels.       
        """
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Make an inference with the network.
        
        Args:
            x - input image (batch x channels x height x width).
        
        Returns:
            (out, mu, logvar) where:
                out - reconstructed image (batch x channels x height x width).
                mu - mean of sample in latent space.
                logvar - log variance of sample in latent space.
        """
        

        mu, logvar = self.encode(x)
        logvar = self.dequant(logvar)
        mu = self.dequant(mu)
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z)

        return out, mu, logvar

    def train_self(self,
                   data_path: str,
                   epochs: int,
                   weights_file: str) -> None:
        """Train the BetaVAE network.
        
        Args:
            data_path - path to training dataset.  This should be a valid
                torch dataset with different classes for each level of each
                partition.
            epochs - number of epochs to train the network.
            weights_file - name of file to save trained weights.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.eval()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_d),
            torchvision.transforms.ToTensor()])
        if self.n_chan == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.input_d),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor()])
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

        for epoch in range(epochs):
            epoch_loss = 0
            for data in train_loader:
                input, _ = data
                input = input.to(device)
                out, mu, logvar = network(input)

                if epoch == 75:
                    for group in optimizer.param_groups:
                        group['lr'] = 1e-6

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=input,
                    size_average=False)
                loss = ce_loss + torch.mul(kl_loss, self.beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            print(f'Epoch: {epoch}; Loss: {loss}')

        print('Training finished, saving weights...')
        torch.save(network, weights_file)

    def train_n_validate(self,
                train_loader: DataLoader,
                val_loader: DataLoader,
                epochs: int,
                weights_file: str) -> None:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
        network.eval()

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

        num_samples = len(train_loader)
        print(f'Num Samples: {num_samples}')

        columns = ['train_ce_loss', 'train_kl_loss', 'train_total_loss',
                   'val_ce_loss', 'val_kl_loss', 'val_total_loss']

        train_results = pd.DataFrame(columns=columns)
        avg_losses = dict.fromkeys(columns)
        
        for epoch in range(epochs):

        ###################
        # train the model #
        ###################

            epoch_ce_loss = 0
            epoch_kl_loss = 0
            epoch_total_loss = 0

            for data in train_loader:
                input, _ = data
                input = input.to(device)
                out, mu, logvar = network(input)

                if epoch == 75:
                    for group in optimizer.param_groups:
                        group['lr'] = 1e-6

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=input,
                    size_average=False)
                loss = ce_loss + torch.mul(kl_loss, self.beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce_loss += ce_loss.item()
                epoch_kl_loss += kl_loss.detach().numpy()
                epoch_total_loss += loss.detach().numpy()
            
            avg_losses['train_ce_loss'] = epoch_ce_loss/num_samples
            avg_losses['train_kl_loss'] = epoch_kl_loss/num_samples
            avg_losses['train_total_loss'] = epoch_total_loss/num_samples

        ######################
        # validate the model #
        ######################

            with torch.no_grad():
                network.eval()  

                epoch_ce_loss = 0
                epoch_kl_loss = 0
                epoch_total_loss = 0

                for data  in val_loader:

                    input, _ = data
                    input = input.to(device)
                    out, mu, logvar = network(input)
                    
                    kl_loss = torch.mul(
                            input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                            other=0.5)
                    ce_loss = torch.nn.functional.binary_cross_entropy(
                            input=out,
                            target=input,
                            size_average=False)
                    loss = ce_loss + torch.mul(kl_loss, self.beta)

                    epoch_ce_loss += ce_loss.item()
                    epoch_kl_loss += kl_loss.detach().numpy()
                    epoch_total_loss += loss.detach().numpy()
                
                avg_losses['val_ce_loss'] = epoch_ce_loss/num_samples
                avg_losses['val_kl_loss'] = epoch_kl_loss/num_samples
                avg_losses['val_total_loss'] = epoch_total_loss/num_samples
                
                train_results = train_results.append(avg_losses, ignore_index=True)
    
                print(f"Epoch: {epoch}; Train_loss: {avg_losses['train_total_loss']}; Val_loss: {avg_losses['val_total_loss']}")

        print('Training finished, saving results csv & saving weights...')

        train_results.to_csv('results/out.csv', index=False)
        torch.save(network.state_dict(), weights_file)
 


class QuantEncoder(torch.nn.Module):
    """Encoder-only portion of the BetaVAE OOD detector.  This calss is not
    trainable.
    
    Args:
        n_latent - size of latent space (encoder output)
        n_chan - number of channels in the input image
        input_d - size of input images (height x width)
    """

    def __init__(self,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int]) -> None:
        super(QuantEncoder, self).__init__()

        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        self.enc_conv1 = torch.nn.Conv2d(
            in_channels=self.n_chan,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv1_bn = torch.nn.BatchNorm2d(128)
        self.enc_conv1_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv1_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv2_bn = torch.nn.BatchNorm2d(64)
        self.enc_conv2_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv2_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv3_bn = torch.nn.BatchNorm2d(32)
        self.enc_conv3_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv3_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True, ceil_mode=True)

        self.enc_conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv4_bn = torch.nn.BatchNorm2d(16)
        self.enc_conv4_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv4_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        self.enc_dense1_af = torch.nn.LeakyReLU(0.1)
        
        self.enc_dense2 = torch.nn.Linear(2048, 1000)
        self.enc_dense2_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4mu_af = torch.nn.LeakyReLU(0.1)
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = torch.nn.LeakyReLU(0.1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode tensor x to its latent representation.
        
        Args:
            x - batch x channels x height x width tensor.

        Returns:
            (mu, var) where mu is sample mean and var is log variance in
            latent space.
        """
        x = self.quant(x)
        z = x
        z = self.enc_conv1(z)
        z = self.enc_conv1_bn(z)
        z = self.enc_conv1_af(z)

        z = self.dequant(z)
        z, self.indices1 = self.enc_conv1_pool(z)
        z = self.quant(z)

        z = self.enc_conv2(z)
        z = self.enc_conv2_bn(z)
        z = self.enc_conv2_af(z)

        z = self.dequant(z)
        z, self.indices2 = self.enc_conv2_pool(z)
        z = self.quant(z)

        z = self.enc_conv3(z)
        z = self.enc_conv3_bn(z)
        z = self.enc_conv3_af(z)

        z = self.dequant(z)
        z, self.indices3 = self.enc_conv3_pool(z)
        z = self.quant(z)

        z = self.enc_conv4(z)
        z = self.enc_conv4_bn(z)
        z = self.enc_conv4_af(z)

        z = self.dequant(z)
        z, self.indices4 = self.enc_conv4_pool(z)
        z = self.quant(z)

        z = z.reshape(z.size(0), -1)
        # z = z.view(z.size(0), -1)
        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        var = self.enc_dense4var(z)
        var = self.enc_dense4var_af(var)

        z = self.dequant(z)
        mu = self.dequant(mu)
        var = self.dequant(var)

        return mu, var

    def get_layer_size(self, layer: int) -> Tuple[int]:
        """Given a network with some input size, calculate the dimensions of
        the resulting layers.
        
        Args:
            layer - layer number (for the encoder: 1 -> 2 -> 3 -> 4, for the
                decoder: 4 -> 3 -> 2 -> 1).

        Returns:
            (y, x) where y is the layer height in pixels and x is the layer
            width in pixels.       
        """
        y_l, x_l = self.input_d
        for i in range(layer - 1):
            y_l = math.ceil((y_l - 2) / 2 + 1)
            x_l = math.ceil((x_l - 2) / 2 + 1)
        return y_l, x_l

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Make an inference with the network.
        
        Args:
            x - input image (batch x channels x height x width).
        
        Returns:
            (mu, logvar) where mu is mean and logvar is the log varinace of
            latent space representation of input sample x.
        """
        return self.encode(x)