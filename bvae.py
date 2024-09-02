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

class BetaVae(torch.nn.Module):
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
        super(BetaVae, self).__init__()
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

        # print(z)
        # print(z.size(0))

        z = z.reshape(z.size(0), -1)
        # z = z.view(z.size(0), -1)

        self.times['enc_dense1'] = time.time()

        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        self.times['enc_dense1'] = time.time() - self.times['enc_dense1']   

        self.times['enc_dense2'] = time.time()

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        self.times['enc_dense2'] = time.time() - self.times['enc_dense2']

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

        self.times['dec_dense2'] = time.time()

        y = self.dec_dense2(y)
        y = self.dec_dense2_af(y)

        self.times['dec_dense2'] = time.time() - self.times['dec_dense2']


        self.times['dec_dense1'] = time.time()

        y = self.dec_dense1(y)
        y = self.dec_dense1_af(y)

        self.times['dec_dense1'] = time.time() - self.times['dec_dense1']


        y = torch.reshape(y, [self.batch, 16, self.y_5, self.x_5])

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
                weights_file: str,
                lr: float,
                file_prefix: str = '',
                early_stop_patience: int = 10000) -> None:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = self.to(device)
       

        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        num_train_samples = len(train_loader)
        print(f'Num Train Samples: {num_train_samples}')

        columns = ['train_ce_loss', 'train_kl_loss', 'train_total_loss',
                   'val_ce_loss', 'val_kl_loss', 'val_total_loss']

        train_results = pd.DataFrame(columns=columns)
        avg_losses = dict.fromkeys(columns)
        
        for epoch in range(epochs):

        ###################
        # train the model #
        ###################
            print("[Epoch:{}/{}]".format(epoch+1, epochs))
            epoch_ce_loss = 0
            epoch_kl_loss = 0
            epoch_total_loss = 0

            for i, data in enumerate(train_loader):
                print("[Batch:{}/{}]".format(i+1, len(train_loader)))#, end='\r')
                input, _ = data
                input = input.to(device)
                out, mu, logvar = network(input)

                # print(input)
                # print(out)
                # return

                if epoch == 75:
                    for group in optimizer.param_groups:
                        group['lr'] = 1e-6

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=input)#,
                    # size_average=False)
                
                loss = ce_loss + torch.mul(kl_loss, self.beta)

                # print("CE LOSS: ", ce_loss.item())
                # print("KL LOSS: ", kl_loss.item())
                # print("TOTAL LOSS: ", loss.item())
                # return 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce_loss += ce_loss.item()
                epoch_kl_loss += kl_loss.detach().numpy()
                epoch_total_loss += loss.detach().numpy()

                break
            
            avg_losses['train_ce_loss'] = epoch_ce_loss/num_train_samples
            avg_losses['train_kl_loss'] = epoch_kl_loss/num_train_samples
            avg_losses['train_total_loss'] = epoch_total_loss/num_train_samples

        ######################
        # validate the model #
        ######################

            num_val_samples = len(val_loader)

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
                            target=input)#,
                            # size_average=False)
                    loss = ce_loss + torch.mul(kl_loss, self.beta)

                    epoch_ce_loss += ce_loss.item()
                    epoch_kl_loss += kl_loss.detach().numpy()
                    epoch_total_loss += loss.detach().numpy()

                    break
                
                avg_losses['val_ce_loss'] = epoch_ce_loss/num_val_samples
                avg_losses['val_kl_loss'] = epoch_kl_loss/num_val_samples
                avg_losses['val_total_loss'] = epoch_total_loss/num_val_samples
                
                train_results = train_results.append(avg_losses, ignore_index=True)
    
                print(f"Epoch: {epoch}; Train_loss: {avg_losses['train_total_loss']}; Val_loss: {avg_losses['val_total_loss']}")

        print('Training finished, saving results csv & saving weights...')

        train_results.to_csv('/Users/aditya/Large Files/NEW FYP/TRAINED_MODELS/CSV_FILES'+file_prefix+'out.csv', index=False)
        torch.save(network.state_dict(), weights_file)
 


class Encoder(torch.nn.Module):
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
        super(Encoder, self).__init__()

        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16
        
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or convert a BetaVAE model.')
    parser.add_argument(
        'action',
        choices=['train', 'convert'],
        metavar='ACTION',
        help='Train a new network or convert one to encoder-only.')
    parser.add_argument(
        '--weights',
        help='Path to weights file (initial weights to use if the train '
             'option is selected).')
    parser.add_argument(
        '--beta',
        help='Beta value for training.')
    parser.add_argument(
        '--n_latent',
        help='Number of latent variables in the model')
    parser.add_argument(
        '--dimensions',
        help='Dimension of input image accepted by the network (height x '
             'width).')
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Network accepts grayscale images if this flag is selected.')
    parser.add_argument(
        '--dataset',
        help='Path to dataset.  This data set should be a folder containing '
             'subfolders for level of variation for each partition.')
    parser.add_argument(
        '--batch',
        help='Batch size to use for training.')
    args = parser.parse_args()

    if args.action == 'train':
        if not (args.beta and args.n_latent and args.dimensions and 
                args.dataset and args.batch):
            print('The optional arguments "--beta", "--n_latent", "--batch" '
                  '"--dimensions", and "--dataset" are required for training')
            sys.exit(1)
        n_latent = int(args.n_latent)
        beta = float(args.beta)
        input_dimensions = tuple([int(i) for i in args.dimensions.split('x')])
        batch = int(args.batch)
        print(f'Starting training for input size {args.dimensions}')
        print(f'beta={beta}')
        print(f'n_latent={n_latent}')
        print(f'batch={batch}')
        print(f'Using data set {args.dataset}')
        network = BetaVae(
            n_latent,
            beta,
            n_chan=1 if args.grayscale else 3,
            input_d=input_dimensions,
            batch=batch)
        network.train_self(
            data_path=args.dataset,
            epochs=100,
            weights_file=f'bvae_n{n_latent}_b{beta}_'
                         f'{"bw" if args.grayscale else ""}_'
                         f'{"x".join([str(i) for i in input_dimensions])}.pt')

    elif args.action == 'convert':
        if not args.weights:
            print('The optional argument "--weights" is required for model '
                  'conversion.')
            sys.exit(1)
        print(f'Converting model {args.weights} to encoder-state-dict-only '
              f'version...')
        full_model = torch.load(args.weights)
        encoder = Encoder(
            n_latent=full_model.n_latent,
            n_chan=full_model.n_chan,
            input_d=full_model.input_d)
        full_dict = full_model.state_dict()
        encoder_dict = encoder.state_dict()
        for key in encoder_dict:
            encoder_dict[key] = full_dict[key]
        torch.save(encoder_dict, f'enc_only_{args.weights}')
