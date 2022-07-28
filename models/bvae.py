#!/usr/bin/env python3
"""BetaVAE OOD detector module.  This module contains everything needed to
train the BetaVAE OOD detector and convert the trained model to an encoder-
only model for efficient test-time execution."""


from typing import Callable, Tuple
import argparse
import math
import sys
import numpy
import scipy.stats
import torch
import torchvision
import PIL


class BetaVae(torch.nn.Module):
    """BetaVAE OOD detector.  This class includes both the encoder and
    decoder portions of the model.
    
    Args:
        n_latent - number of latent dimensions
        beta - hyperparameter beta to use during training
        n_chan - number of channels in the input image
        input_d - height x width tuple of input image size in pixels
        activation - activation function to use for all hidden layers
        head2logvar - 2nd distribution parameter learned by encoder:
            'logvar', 'logvar+1', 'var', or 'neglogvar'.
        interpolation - PIL image interpolation method on resize
    """

    def __init__(self,
                 n_latent: int,
                 beta: float,
                 n_chan: int,
                 input_d: Tuple[int],
                 batch: int = 1,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 head2logvar: str = 'logvar',
                 interpolation: int = PIL.Image.BILINEAR) -> None:
        super(BetaVae, self).__init__()
        self.batch = batch
        self.n_latent = n_latent
        self.beta = beta
        self.n_chan = n_chan
        self.input_d = input_d
        self.interpolation = int(interpolation)
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
            padding='same')
        self.enc_conv1_bn = torch.nn.BatchNorm2d(128)
        self.enc_conv1_af = activation
        self.enc_conv1_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv2_bn = torch.nn.BatchNorm2d(64)
        self.enc_conv2_af = activation
        self.enc_conv2_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv3_bn = torch.nn.BatchNorm2d(32)
        self.enc_conv3_af = activation
        self.enc_conv3_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv4_bn = torch.nn.BatchNorm2d(16)
        self.enc_conv4_af = activation
        self.enc_conv4_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        self.enc_dense1_af = activation
        
        self.enc_dense2 = torch.nn.Linear(2048, 1000)
        self.enc_dense2_af = activation

        self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3_af = activation

        self.enc_dense4mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4mu_af = activation
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = activation
        self.enc_head2logvar = self.Head2LogVar(head2logvar)

        self.dec_dense4 = torch.nn.Linear(self.n_latent, 250)
        self.dec_dense4_af = activation
        
        self.dec_dense3 = torch.nn.Linear(250, 1000)
        self.dec_dense3_af = activation

        self.dec_dense2 = torch.nn.Linear(1000, 2048)
        self.dec_dense2_af = activation

        self.dec_dense1 = torch.nn.Linear(2048, self.hidden_units)
        self.dec_dense1_af = activation

        self.dec_conv4_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv4 = torch.nn.ConvTranspose2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv4_bn = torch.nn.BatchNorm2d(32)
        self.dec_conv4_af = activation

        self.dec_conv3_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv3 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv3_bn = torch.nn.BatchNorm2d(64)
        self.dec_conv3_af = activation

        self.dec_conv2_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv2 = torch.nn.ConvTranspose2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1)
        self.dec_conv2_bn = torch.nn.BatchNorm2d(128)
        self.dec_conv2_af = activation

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

        z = z.view(z.size(0), -1)
        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        pvar = self.enc_dense4var(z)
        pvar = self.enc_dense4var_af(pvar)
        logvar = self.enc_head2logvar(pvar)

        return mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation to generate a reconstructed image.

        Args:
            z - 1 x n_latent input tensor.

        Returns:
            A batch x channels x height x width tensor representing the
            reconstructed image.
        """
        y = self.dec_dense4(z)
        y = self.dec_dense4_af(y)

        y = self.dec_dense3(y)
        y = self.dec_dense3_af(y)

        y = self.dec_dense2(y)
        y = self.dec_dense2_af(y)

        y = self.dec_dense1(y)
        y = self.dec_dense1_af(y)

        y = torch.reshape(y, [self.batch, 16, self.y_5, self.x_5])
        y = self.dec_conv4_pool(
            y,
            self.indices4,
            output_size=torch.Size([self.batch, 16, self.y_4, self.x_4]))
        y = self.dec_conv4(y)
        y = self.dec_conv4_bn(y)
        y = self.dec_conv4_af(y)

        y = self.dec_conv3_pool(
            y,
            self.indices3,
            output_size=torch.Size([self.batch, 32, self.y_3, self.x_3]))
        y = self.dec_conv3(y)
        y = self.dec_conv3_bn(y)
        y = self.dec_conv3_af(y)

        y = self.dec_conv2_pool(
            y,
            self.indices2,
            output_size=torch.Size([self.batch, 64, self.y_2, self.x_2]))
        y = self.dec_conv2(y)
        y = self.dec_conv2_bn(y)
        y = self.dec_conv2_af(y)

        y = self.dec_conv1_pool(
            y,
            self.indices1,
            output_size=torch.Size([self.batch, 128, self.input_d[0], self.input_d[1]]))
        y = self.dec_conv1(y)
        y = self.dec_conv1_bn(y)
        y = self.dec_conv1_af(y)
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
        """Train the BetaVAE network.  The learning rate is hardcoded based on
        the original BetaVAE OOD detection paper.  This training method also
        forces the use of a manual seed to ensure repeatability.
        
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

        torch.manual_seed(0)
        numpy.random.seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.input_d, interpolation=self.interpolation)])
        if self.n_chan == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(self.input_d, interpolation=self.interpolation),
                torchvision.transforms.Grayscale()])
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
                    reduction='sum')
                loss = ce_loss + torch.mul(kl_loss, self.beta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss
            print(f'Epoch: {epoch}; Loss: {loss}')

        print('Training finished, saving weights...')
        torch.save(network, weights_file)

    def mig(self, data_path: str, iters: int, samples: int = 100) -> float:
        """Find this network's mutual information gain on a given data set.
        
        Args:
            data_path - path to data set of images to on which to calculate
                mutual information gain.
            iters - number of iterations to sample latent space.  Higher gives
                better accuracy at the expence of more time.
                
        Returns:
            Mutual information gain of this network.
        """
        # 1. Inference on the Network to get Latent Dists
        original_batch = self.batch
        self.batch = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = self.to(device)
        network.eval()

        torch.manual_seed(0)
        numpy.random.seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224))])
        dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms)
        cv_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True)
        mu = numpy.zeros((self.n_latent, len(dataset.imgs)))
        logvar = numpy.zeros((self.n_latent, len(dataset.imgs)))
        f_mask = numpy.zeros(len(dataset.imgs))
        f_counts = {f: 0 for _, f in dataset.class_to_idx.items()}
        idx = 0
        for data in cv_loader:
            x, partition = data
            x = x.to(device)
            _, m, lv = network(x)
            mu[:, idx] = m.detach().cpu().numpy()
            logvar[:, idx] = lv.detach().cpu().numpy()
            f_mask[idx] = partition
            f_counts[int(partition)] += 1
            idx += 1

        f_probs = [count / len(dataset.imgs) for _, count in f_counts.items()]
        f_entropy = scipy.stats.entropy(f_probs)
        migs = numpy.zeros(iters)
        for i in range(iters):
            print(f'Getting MIG for Iter {i}')
            std = numpy.exp(logvar / 2)
            smp = numpy.zeros((self.n_latent, len(dataset.imgs), samples))
            # Get samples for all images latent dists
            for s in range(samples):
                eps = numpy.random.randn(*std.shape)
                smp[:, :, s] = mu + std * eps

            # Get probability of each sample occurring
            p_lat = numpy.zeros((self.n_latent, len(dataset.imgs), samples))
            for lat in range(self.n_latent):
                for d in range(len(dataset.imgs)):
                    sig = numpy.exp(logvar[lat, d] / 2)
                    p_lat[lat, d, :] = (1 / (sig * numpy.sqrt(2 * numpy.pi))) \
                            * numpy.exp(-0.5 * ((smp[lat, d, :] - mu[lat, d]) \
                            / sig) ** 2)

            h_lat = numpy.zeros(self.n_latent)
            # Get the entropies of each latent variable across whole input set
            for lat in range(self.n_latent):
                h_lat[lat] = scipy.stats.entropy(p_lat[lat, :, :].flatten())
                mig = 0
            for f in f_counts:
                h_lat_given_f = numpy.zeros(self.n_latent)
                class_length = 0
                for lat in range(self.n_latent):
                    p_lat_given_f = numpy.zeros((f_counts[f], samples))
                    idx = 0
                    for d in range(len(dataset.imgs)):
                        if f_mask[d] == f:
                            p_lat_given_f[idx, :] = p_lat[lat, d, :]
                            idx += 1
                    h_lat_given_f[lat] = scipy.stats.entropy(p_lat_given_f.flatten())
                mi = h_lat - h_lat_given_f
                mi.sort()
                mig += (mi[-1] - mi[-2]) / f_entropy
            migs[i] = 1 / len(dataset.classes) * mig
        self.batch = original_batch
        return numpy.mean(migs)

    class Head2LogVar:
        """This class defines the final layer on one of the encoder heads.
        Essentially it performs an element-wise operation on the output of
        each neuron in the preceding layer in order to transform the input
        to log(var).

        Args:
            logvar - transform from what to logvar: 'logvar', 'logvar+1',
                'neglogvar', or 'var'.
        """

        def __init__(self, type: str = 'logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x: torch.Tensor):
            """IF x == log(sig^2):
            THEN x = log(sig^2)"""
            return x

        def logvarplusone(self, x: torch.Tensor):
            """IF x = log(sig^2 + 1)
            THEN log(e^x - 1) = log(sig^2)"""
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x: torch.Tensor):
            """IF x = -log(sig^2)
            THEN -x = log(sig^2)"""
            return x.neg()

        def var(self, x: torch.Tensor):
            """IF x = sig^2
            THEN log(x) = log(sig^2)"""
            return x.add(self.eps).log()

        def __call__(self, input: torch.Tensor):
            """Runs when calling instance of object."""
            return self.type(input)


class Encoder(torch.nn.Module):
    """Encoder-only portion of the BetaVAE OOD detector.  This class is not
    trainable, weights must be pulled from a trained BetaVae class.
    
    Args:
        n_latent - size of latent space (encoder output)
        n_chan - number of channels in the input image
        input_d - size of input images (height x width)
        activation - activation function for all layers
    """

    def __init__(self,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int],
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 head2logvar: str = 'logvar') -> None:
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
            padding='same')
        self.enc_conv1_bn = torch.nn.BatchNorm2d(128)
        self.enc_conv1_af = activation
        self.enc_conv1_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv2_bn = torch.nn.BatchNorm2d(64)
        self.enc_conv2_af = activation
        self.enc_conv2_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv3_bn = torch.nn.BatchNorm2d(32)
        self.enc_conv3_af = activation
        self.enc_conv3_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True, ceil_mode=True)

        self.enc_conv4 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            bias=False,
            padding='same')
        self.enc_conv4_bn = torch.nn.BatchNorm2d(16)
        self.enc_conv4_af = activation
        self.enc_conv4_pool = torch.nn.MaxPool2d(
            kernel_size=2,
            return_indices=True,
            ceil_mode=True)

        self.enc_dense1 = torch.nn.Linear(self.hidden_units, 2048)
        self.enc_dense1_af = activation
        
        self.enc_dense2 = torch.nn.Linear(2048, 1000)
        self.enc_dense2_af = activation

        self.enc_dense3 = torch.nn.Linear(1000, 250)
        self.enc_dense3_af = activation

        self.enc_dense4mu = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4mu_af = activation
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = activation
        self.enc_head2logvar = self.Head2LogVar(head2logvar)

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

        z = z.view(z.size(0), -1)
        z = self.enc_dense1(z)
        z = self.enc_dense1_af(z)

        z = self.enc_dense2(z)
        z = self.enc_dense2_af(z)

        z = self.enc_dense3(z)
        z = self.enc_dense3_af(z)

        mu = self.enc_dense4mu(z)
        mu = self.enc_dense4mu_af(mu)

        pvar = self.enc_dense4var(z)
        pvar = self.enc_dense4var_af(pvar)
        logvar = self.enc_head2logvar(pvar)

        return mu, logvar

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

    class Head2LogVar:
        """This class defines the final layer on one of the encoder heads.
        Essentially it performs an element-wise operation on the output of
        each neuron in the preceding layer in order to transform the input
        to log(var).

        Args:
            logvar - transform from what to logvar: 'logvar', 'logvar+1',
                'neglogvar', or 'var'.
        """

        def __init__(self, type: str = 'logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x: torch.Tensor):
            """IF x == log(sig^2):
            THEN x = log(sig^2)"""
            return x

        def logvarplusone(self, x: torch.Tensor):
            """IF x = log(sig^2 + 1)
            THEN log(e^x - 1) = log(sig^2)"""
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x: torch.Tensor):
            """IF x = -log(sig^2)
            THEN -x = log(sig^2)"""
            return x.neg()

        def var(self, x: torch.Tensor):
            """IF x = sig^2
            THEN log(x) = log(sig^2)"""
            return x.add(self.eps).log()

        def __call__(self, input: torch.Tensor):
            """Runs when calling instance of object."""
            return self.type(input)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or convert a BetaVAE model.')
    parser.add_argument(
        'action',
        choices=['train', 'convert', 'mig'],
        metavar='ACTION',
        help='TRAIN a new network, CONVERT a pretrained network to '
             'encoder-only, or find the MIG of a given network on a '
             'particular dataset')
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
    parser.add_argument(
        '--activation',
        choices=['leaky', 'relu', 'tanh'],
        default='relu',
        help='Activation function for hidden layers')
    parser.add_argument(
        '--head2learnwhat',
        choices=['logvar', 'logvar+1', 'neglogvar', 'var'],
        default='logvar',
        help='What the 2nd encoder head learns')
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs to train for.')
    parser.add_argument(
        '--interpolation',
        default='bilinear',
        choices=['nearest', 'bilinear', 'bicubic'],
        help='Interpolation to use on resize')
    args = parser.parse_args()

    torch.manual_seed(0)
    numpy.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(0)

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
        activation = {
            'relu': torch.nn.ReLU(),
            'leaky': torch.nn.LeakyReLU(0.1),
            'tanh': torch.nn.Tanh()}[args.activation]
        interpolation = {
            'nearest': PIL.Image.NEAREST,
            'bilinear': PIL.Image.BILINEAR,
            'bicubic': PIL.Image.BICUBIC}[args.interpolation]
        print(f'Starting training for input size {args.dimensions}')
        print(f'beta={beta}')
        print(f'n_latent={n_latent}')
        print(f'batch={batch}')
        print(f'epochs={args.epochs}')
        print(f'activation function={args.activation}')
        print(f'Encoder learns mean and {args.head2learnwhat}')
        print(f'Using data set {args.dataset}')
        print(f'Using {interpolation} interpolation')
        network = BetaVae(
            n_latent,
            beta,
            n_chan=1 if args.grayscale else 3,
            input_d=input_dimensions,
            batch=batch,
            activation=activation,
            head2logvar=args.head2learnwhat,
            interpolation=interpolation)
        network.train_self(
            data_path=args.dataset,
            epochs=args.epochs,
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
    elif args.action == 'mig':
        if not args.weights or not args.dataset:
            print('The optional arguments "--weights" and "--dataset" are '
                  'required for MIG calculation.')
        model = torch.load(args.weights)
        print(f'Calculating MIG for model {args.weights}...')
        mig = model.mig(args.dataset, 5)
        print(f'MIG = {mig} nats')
