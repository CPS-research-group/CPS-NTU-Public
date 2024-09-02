#!/usr/bin/env python3
"""BetaVAE OOD detector module.  This module contains everything needed to
train the BetaVAE OOD detector and convert the trained model to an encoder-
only model for efficient test-time execution."""

from typing import Tuple
import argparse
import math
import sys
import torch
import torchvision
import pandas as pd
from torch.utils.data import DataLoader
import time

class ModBetaVae(torch.nn.Module):
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
        super(ModBetaVae, self).__init__()
        self.batch = batch
        self.n_latent = n_latent
        self.beta = beta
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.y_6, self.x_6 = self.get_layer_size(6)
        self.hidden_units = self.y_6 * self.x_6 * 8

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

        self.enc_conv5 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv5_bn = torch.nn.BatchNorm2d(8)
        self.enc_conv5_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv5_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        # self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        # self.enc_dense1_af = torch.nn.LeakyReLU(0.1)
        
        # self.enc_dense2 = torch.nn.Linear(2048, 1000)
        # self.enc_dense2_af = torch.nn.LeakyReLU(0.1)

        # self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3 = torch.nn.Linear(self.hidden_units, 250)
        self.enc_dense3_af = torch.nn.LeakyReLU(0.1)

        self.enc_dense4mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4mu_af = torch.nn.LeakyReLU(0.1)
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = torch.nn.LeakyReLU(0.1)

        self.dec_dense4 = torch.nn.Linear(self.n_latent, 250)
        self.dec_dense4_af = torch.nn.LeakyReLU(0.1)
        
        # self.dec_dense3 = torch.nn.Linear(250, 1000)
        self.dec_dense3 = torch.nn.Linear(250, self.hidden_units)
        self.dec_dense3_af = torch.nn.LeakyReLU(0.1)

        # self.dec_dense2 = torch.nn.Linear(1000, 2048)
        # self.dec_dense2_af = torch.nn.LeakyReLU(0.1)

        # self.dec_dense1 = torch.nn.Linear(2048, self.hidden_units)
        # self.dec_dense1_af = torch.nn.LeakyReLU(0.1)

        self.dec_conv5_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv5 = torch.nn.ConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv5_bn = torch.nn.BatchNorm2d(16)
        self.dec_conv5_af = torch.nn.LeakyReLU(0.1)

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

        self.times = {}

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode tensor x to its latent representation.
        
        Args:
            x - batch x channels x height x width tensor.

        Returns:
            (mu, var) where mu is sample mean and var is log variance in
            latent space.
        """

        z = x
        
        self.times['enc_conv1'] = time.time()

        z = self.enc_conv1(z)
        z = self.enc_conv1_bn(z)
        z = self.enc_conv1_af(z)
        z, self.indices1 = self.enc_conv1_pool(z)
      
        self.times['enc_conv1'] = time.time() - self.times['enc_conv1']

        self.times['enc_conv2'] = time.time()

        z = self.enc_conv2(z)
        z = self.enc_conv2_bn(z)
        z = self.enc_conv2_af(z)

        z, self.indices2 = self.enc_conv2_pool(z)
      
        self.times['enc_conv2'] = time.time() - self.times['enc_conv2']


        self.times['enc_conv3'] = time.time()
        z = self.enc_conv3(z)
        z = self.enc_conv3_bn(z)
        z = self.enc_conv3_af(z)

        z, self.indices3 = self.enc_conv3_pool(z)

        self.times['enc_conv3'] = time.time() - self.times['enc_conv3']


        self.times['enc_conv4'] = time.time()
        z = self.enc_conv4(z)
        z = self.enc_conv4_bn(z)
        z = self.enc_conv4_af(z)

        z, self.indices4 = self.enc_conv4_pool(z)

        self.times['enc_conv4'] = time.time() - self.times['enc_conv4']

        self.times['enc_conv5'] = time.time()
        z = self.enc_conv5(z)
        z = self.enc_conv5_bn(z)
        z = self.enc_conv5_af(z)

        z, self.indices5 = self.enc_conv5_pool(z)

        self.times['enc_conv5'] = time.time() - self.times['enc_conv5']

        # print(z)
        # print(z.size(0))

        z = z.reshape(z.size(0), -1)
        # z = z.view(z.size(0), -1)

        # self.times['enc_dense1'] = time.time()

        # z = self.enc_dense1(z)
        # z = self.enc_dense1_af(z)

        # self.times['enc_dense1'] = time.time() - self.times['enc_dense1']   

        # self.times['enc_dense2'] = time.time()

        # z = self.enc_dense2(z)
        # z = self.enc_dense2_af(z)

        # self.times['enc_dense2'] = time.time() - self.times['enc_dense2']

        self.times['enc_dense3'] = time.time()

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        self.times['enc_dense3'] = time.time() - self.times['enc_dense3']

        self.times['enc_dense4'] = time.time()

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        var = self.enc_dense4var(z)
        var = self.enc_dense4var_af(var)

        self.times['enc_dense4'] = time.time() - self.times['enc_dense4']

        return mu, var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation to generate a reconstructed image.

        Args:
            z - 1 x n_latent input tensor.

        Returns:
            A batch x channels x height x width tensor representing the
            reconstructed image.
        """

        
        self.times['dec_dense4'] = time.time()

        y = self.dec_dense4(z)
        y = self.dec_dense4_af(y)

        self.times['dec_dense4'] = time.time() - self.times['dec_dense4']

        self.times['dec_dense3'] = time.time()

        y = self.dec_dense3(y)
        y = self.dec_dense3_af(y)

        self.times['dec_dense3'] = time.time() - self.times['dec_dense3']

        # self.times['dec_dense2'] = time.time()

        # y = self.dec_dense2(y)
        # y = self.dec_dense2_af(y)

        # self.times['dec_dense2'] = time.time() - self.times['dec_dense2']


        # self.times['dec_dense1'] = time.time()

        # y = self.dec_dense1(y)
        # y = self.dec_dense1_af(y)

        # self.times['dec_dense1'] = time.time() - self.times['dec_dense1']


        y = torch.reshape(y, [self.batch, 8, self.y_6, self.x_6])

        self.times['dec_conv5'] = time.time() 

        y = self.dec_conv5_pool(
            y,
            self.indices5,
            output_size=torch.Size([self.batch, 8, self.y_5, self.x_5]))
     
        
        y = self.dec_conv5(y)
        y = self.dec_conv5_bn(y)
        y = self.dec_conv5_af(y)

        self.times['dec_conv5'] = time.time() - self.times['dec_conv5']

        self.times['dec_conv4'] = time.time() 

        y = self.dec_conv4_pool(
            y,
            self.indices4,
            output_size=torch.Size([self.batch, 16, self.y_4, self.x_4]))
     
        
        y = self.dec_conv4(y)
        y = self.dec_conv4_bn(y)
        y = self.dec_conv4_af(y)

        self.times['dec_conv4'] = time.time() - self.times['dec_conv4']


        self.times['dec_conv3'] = time.time()
   
        y = self.dec_conv3_pool(
            y,
            self.indices3,
            output_size=torch.Size([self.batch, 32, self.y_3, self.x_3]))
   

        y = self.dec_conv3(y)
        y = self.dec_conv3_bn(y)
        y = self.dec_conv3_af(y)

        self.times['dec_conv3'] = time.time() - self.times['dec_conv3']

        
        self.times['dec_conv2'] = time.time()

        y = self.dec_conv2_pool(
            y,
            self.indices2,
            output_size=torch.Size([self.batch, 64, self.y_2, self.x_2]))
   

        y = self.dec_conv2(y)
        y = self.dec_conv2_bn(y)
        y = self.dec_conv2_af(y)

        self.times['dec_conv2'] = time.time() - self.times['dec_conv2']
    
        self.times['dec_conv1'] = time.time()

        y = self.dec_conv1_pool(
            y,
            self.indices1,
            output_size=torch.Size([self.batch, 128, self.input_d[0], self.input_d[1]]))

        y = self.dec_conv1(y)
        y = self.dec_conv1_bn(y)
        y = self.dec_conv1_af(y)

        self.times['dec_conv1'] = time.time() - self.times['dec_conv1']

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
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        z = mu + std * eps
        out = self.decode(z)
        
        return out, mu, logvar



class ModEncoder(torch.nn.Module):
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
        super(ModEncoder, self).__init__()

        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.y_6, self.x_6 = self.get_layer_size(6)
        self.hidden_units = self.y_6 * self.x_6 * 8
        
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

        self.enc_conv5 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            bias=False,
            padding=1)#'same')
        self.enc_conv5_bn = torch.nn.BatchNorm2d(8)
        self.enc_conv5_af = torch.nn.LeakyReLU(0.1)
        self.enc_conv5_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        # self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        # self.enc_dense1_af = torch.nn.LeakyReLU(0.1)
        
        # self.enc_dense2 = torch.nn.Linear(2048, 1000)
        # self.enc_dense2_af = torch.nn.LeakyReLU(0.1)

        # self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3 = torch.nn.Linear(self.hidden_units, 250)
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
       
        z = x
        z = self.enc_conv1(z)
        z = self.enc_conv1_bn(z)
        z = self.enc_conv1_af(z)


        z, self.indices1 = self.enc_conv1_pool(z)

        z = self.enc_conv2(z)
        z = self.enc_conv2_bn(z)
        z = self.enc_conv2_af(z)

    
        z, self.indices2 = self.enc_conv2_pool(z)
    
        z = self.enc_conv3(z)
        z = self.enc_conv3_bn(z)
        z = self.enc_conv3_af(z)

    
        z, self.indices3 = self.enc_conv3_pool(z)
    
        z = self.enc_conv4(z)
        z = self.enc_conv4_bn(z)
        z = self.enc_conv4_af(z)


        z, self.indices4 = self.enc_conv4_pool(z)

        z = self.enc_conv5(z)
        z = self.enc_conv5_bn(z)
        z = self.enc_conv5_af(z)


        z, self.indices5 = self.enc_conv5_pool(z)
    
        z = z.reshape(z.size(0), -1)
        # z = z.view(z.size(0), -1)
        # z = self.enc_dense1(z)
        # z = self.enc_dense1_af(z)

        # z = self.enc_dense2(z)
        # z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        var = self.enc_dense4var(z)
        var = self.enc_dense4var_af(var)

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
