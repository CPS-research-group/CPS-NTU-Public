#!/usr/bin/env python3
"""BetaVAE OOD detector module.  This module contains everything needed to
train the BetaVAE OOD detector and convert the trained model to an encoder-
only model for efficient test-time execution."""

from typing import Callable, Tuple
import argparse
import math
import sys
import numpy
# import scipy.stats
import torch
import torchvision
import PIL

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
except:
    pass


# torch.manual_seed(0)
# numpy.random.seed(0)
# torch.use_deterministic_algorithms(True)
#torch.backends.cudnn.benchmark = False
# torch.cuda.manual_seed(0)

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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
        self.dec_conv4_bn = torch.nn.BatchNorm2d(32)
        self.dec_conv4_af = activation

        self.dec_conv3_pool = torch.nn.MaxUnpool2d(2)
        self.dec_conv3 = torch.nn.ConvTranspose2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=False,
            # padding='same')
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
        """Train the BetaVAE network.

        Args:
            data_path - path to training dataset.  This should be a valid
                torch dataset with different classes for each level of each
                partition.
            epochs - number of epochs to train the network.
            weights_file - name of file to save trained weights.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        network = self.to(device)
        network.eval()

        torch.manual_seed(0)
        numpy.random.seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed(0)

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.input_d),
            torchvision.transforms.ToTensor()])
        if self.n_chan == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.Resize(
                    self.input_d, interpolation=self.interpolation),
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
        original_batch = self.batch
        self.batch = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')
        network = self.to(device)
        network.eval()

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(self.input_d)])
        if self.n_chan == 1:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(
                    self.input_d, interpolation=self.interpolation),
                torchvision.transforms.Grayscale()])
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
            input, partition = data
            input = input.to(device)
            _, m, lv = network(input)
            mu[:, idx] = m.detach().cpu().numpy()
            logvar[:, idx] = lv.detach().cpu().numpy()
            f_mask[idx] = partition
            f_counts[int(partition)] += 1
            idx += 1

        f_probs = [count / len(dataset.imgs) for _, count in f_counts.items()]
        f_entrpy = scipy.stats.entropy(f_probs)
        migs = numpy.zeros(iters)
        for i in range(iters):
            std = numpy.exp(logvar / 2)
            smp = numpy.zeros((self.n_latent, len(dataset.imgs), samples))
            for s in range(samples):
                eps = numpy.random.randn(*std.shape)
                smp[:, :, s] = mu + std * eps
            h_lat = numpy.zeros(self.n_latent)
            for idx, lat in enumerate(range(self.n_latent)):
                p_lat = []
                for d in range(len(dataset.imgs)):
                    sig = numpy.sqrt(numpy.exp(logvar[lat, d]))
                    for s in range(samples):
                        p_lat.append((1 / (sig * numpy.sqrt(2 * numpy.pi))) *
                                     numpy.exp(-0.5 * ((smp[lat, d, s] - mu[lat, d]) / sig) ** 2))
                h_lat[idx] = scipy.stats.entropy(p_lat)
            mig = 0
            for f in dataset.classes:
                h_lat_given_f = numpy.zeros(self.n_latent)
                class_length = 0
                for idx, lat in enumerate(range(self.n_latent)):
                    p_lat_given_f = []
                    for d in range(len(dataset.imgs)):
                        if f_mask[d] == dataset.class_to_idx[f]:
                            sig = numpy.sqrt(numpy.exp(logvar[lat, d]))
                            for s in range(samples):
                                p_lat_given_f.append(
                                    (1 / (sig * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-0.5 * ((smp[lat, d, s] - mu[lat, d]) / sig) ** 2))
                    h_lat_given_f[idx] = scipy.stats.entropy(p_lat_given_f)

                mi = h_lat - h_lat_given_f
                mi.sort()
                mig += (mi[-1] - mi[-2]) / f_entrpy
            migs[i] = 1 / len(dataset.classes) * mig
        self.batch = original_batch
        return numpy.mean(migs)

    class Head2LogVar:
        def __init__(self, type='logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x):
            return x

        def logvarplusone(self, x):
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x):
            return x.neg()

        def var(self, x):
            if isinstance(x, numpy.ndarray):
                return numpy.log(numpy.add(x, self.eps))
            else:
                return x.add(self.eps).log()

        def __call__(self, input):
            return self.type(input)


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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
            # padding='same')
            padding=1)
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
        def __init__(self, type='logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x):
            return x

        def logvarplusone(self, x):
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x):
            return x.neg()

        def var(self, x):
            if isinstance(x, numpy.ndarray):
                return numpy.log(numpy.add(x, self.eps))
            else:
                return x.add(self.eps).log()

        def __call__(self, input):
            return self.type(input)


class QuantizedEncoder(torch.nn.Module):
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
                 input_d: Tuple[int],
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 head2logvar: str = 'logvar') -> None:
        super(QuantizedEncoder, self).__init__()
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.y_2, self.x_2 = self.get_layer_size(2)
        self.y_3, self.x_3 = self.get_layer_size(3)
        self.y_4, self.x_4 = self.get_layer_size(4)
        self.y_5, self.x_5 = self.get_layer_size(5)
        self.hidden_units = self.y_5 * self.x_5 * 16

        self.quant = torch.quantization.QuantStub()
        self.enc_conv1 = torch.nn.Conv2d(
            in_channels=self.n_chan,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=(1, 1))
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
            padding=(1, 1))
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
            padding=(1, 1))
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
            padding=(1, 1))
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
        self.dequant_mu = torch.quantization.DeQuantStub()
        self.enc_dense4var = torch.nn.Linear(250, self.n_latent)
        self.enc_dense4var_af = activation
        self.dequant_var = torch.quantization.DeQuantStub()
        self.enc_head2logvar = self.Head2LogVar(head2logvar)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Encode tensor x to its latent representation.

        Args:
            x - batch x channels x height x width tensor.

        Returns:
            (mu, var) where mu is sample mean and var is log variance in
            latent space.
        """
        z = self.quant(x)
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
        mu = self.dequant_mu(mu)

        pvar = self.enc_dense4var(z)
        pvar = self.enc_dense4var_af(pvar)
        pvar = self.dequant_var(pvar)
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
        def __init__(self, type='logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x):
            return x

        def logvarplusone(self, x):
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x):
            return x.neg()

        def var(self, x):
            if isinstance(x, numpy.ndarray):
                return numpy.log(numpy.add(x, self.eps))
            else:
                return x.add(self.eps).log()

        def __call__(self, input):
            return self.type(input)


class Encoder_TPU(torch.nn.Module):

    def __init__(self,
                 interpreter,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int],
                 interpolation,
                 head2logvar: str = 'logvar'):
        super(Encoder_TPU, self).__init__()
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.interpreter = interpreter
        self.interpolation = interpolation
        self.minvar = 0
        # self.interpreter = edgetpu.make_interpreter(model)
        # self.interpreter.allocate_tensors()
        (self.quant_scale, self.quant_zero_point) = self.interpreter.get_input_details()[
            0]['quantization']
        # print(F"INPUT Quant Details {self.interpreter.get_input_details()}")
        # Check index for the output accordingly.
        (self.quant_scale_out1, self.quant_zero_point_out1) = self.interpreter.get_output_details()[
            1]['quantization']
        (self.quant_scale_out2, self.quant_zero_point_out2) = self.interpreter.get_output_details()[
            0]['quantization']
        # print(F"Output Quant Details {self.interpreter.get_output_details()}")

        self.enc_head2logvar = self.Head2LogVar(head2logvar)

    def callInterpreter_double(self, data, interpreter, outputIndex1, outputIndex2):
        # print(interpreter.get_input_details())
        # print(interpreter.get_output_details())
        common.set_input(interpreter, data)
        interpreter.invoke()
        return interpreter.get_tensor(outputIndex1), interpreter.get_tensor(outputIndex2)

    class Head2LogVar:
        def __init__(self, type='logvar'):
            self.eps = 1e-6
            print(f'eps: {self.eps}')
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x):
            return x

        def logvarplusone(self, x):
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x):
            return x.neg()

        def var(self, x):
            if isinstance(x, numpy.ndarray):
                return numpy.log(numpy.add(x, self.eps))
            else:
                return x.add(self.eps).log()

        def __call__(self, input):
            return self.type(input)

    def encode(self, x):
        # Convert CHW to HWC
        intermediate = torch.moveaxis(x, 0, 2)
        # intermediate = x.permute((1, 2, 0))
        # intermediate = numpy.transpose(x, (1, 2, 0))
        intermediate = (intermediate / self.quant_scale) + \
            (self.quant_zero_point)
        input_to_tpu = numpy.zeros(
            (1, self.input_d[0], self.input_d[1], self.n_chan))
        input_to_tpu[0, :, :, :] = intermediate[:, :, :]
        # Original Mu and logvar
        # mu, logvar = self.callInterpreter_double(intermediate, self.interpreter, 1, 2)
        # Swap Mu and logvar
        mu, logvar = self.callInterpreter_double(
            input_to_tpu, self.interpreter, 2, 1)
        mu = numpy.squeeze(mu)
        logvar = numpy.squeeze(logvar)
        mu.astype(float)
        logvar.astype(float)
        mu = (mu - float(self.quant_zero_point_out1)) * self.quant_scale_out1
        logvar = (logvar - float(self.quant_zero_point_out2)) * self.quant_scale_out2
        # currentvar = numpy.min(logvar)
        # if currentvar < self.minvar:
        #     self.minvar = currentvar
        #     print(f'MIN LOGVAR: {self.minvar}')
        logvar = self.enc_head2logvar(logvar)
        d_kl = 0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)
        # print(d_kl)
        return d_kl

class Encoder_TFLITE_BVAE(torch.nn.Module):
    """TFLITE VERSION Encoder-only portion of the BetaVAE OOD detector.

    Args:
        n_latent - size of latent space (encoder output)
        n_chan - number of channels in the input image
        input_d - size of input images (height x width)
    """

    def __init__(self,
                 interpreter,
                 n_latent: int,
                 n_chan: int,
                 input_d: Tuple[int],
                 interpolation: int = PIL.Image.BILINEAR,
                 activation: torch.nn.Module = torch.nn.ReLU(),
                 head2logvar: str = 'logvar') -> None:
        super(Encoder_TFLITE_BVAE, self).__init__()
        self.n_latent = n_latent
        self.n_chan = n_chan
        self.input_d = input_d
        self.interpreter = interpreter
        self.interpolation = int(interpolation)
        # NEED TO FOLLOW MODEL VALUES
        self.input_details = self.interpreter.get_input_details()
        # print(self.input_details)
        self.output_details = self.interpreter.get_output_details()
        # print(self.output_details)
        (self.quant_scale, self.quant_zero_point) = interpreter.get_input_details()[
            0]['quantization']
        # StatePartitionedCall:1
        (self.quant_scale_out1, self.quant_zero_point_out1) = interpreter.get_output_details()[
            1]['quantization']
        # StatePartitionedCall:0
        (self.quant_scale_out2, self.quant_zero_point_out2) = interpreter.get_output_details()[
            0]['quantization']

        self.enc_head2logvar = self.Head2LogVar(head2logvar)

    class Head2LogVar:
        def __init__(self, type='logvar'):
            self.eps = 1e-6
            self.type = {
                'logvar': self.logvar,
                'logvar+1': self.logvarplusone,
                'neglogvar': self.neglogvar,
                'var': self.var}[type]

        def logvar(self, x):
            return x

        def logvarplusone(self, x):
            return x.exp().add(-1 + self.eps).log()

        def neglogvar(self, x):
            return x.neg()

        def var(self, x):
            if isinstance(x, numpy.ndarray):
                return numpy.log(numpy.add(x, self.eps))
            else:
                return x.add(self.eps).log()

        def __call__(self, input):
            return self.type(input)

    def encode(self, data):
        """Encode tensor data to its latent representation.

        Args:
            data - channels x height x width tensor.

        Returns:
            (mu, var) where mu is sample mean and var is log variance in
            latent space.
        """
        # Convert CHW to HWC
        intermediate = torch.moveaxis(data, 0, 2)
        # Convert real_value flow to int8_value
        intermediate = (intermediate / self.quant_scale) + (self.quant_zero_point)
        intermediate = intermediate.type(torch.int8)
        input_to_tpu = numpy.zeros(
            (1, self.input_d[0], self.input_d[1], self.n_chan), dtype=numpy.int8)
        input_to_tpu[0, :, :, :] = intermediate[:, :, :]
        # mu, logvar = callInterpreter_double(
            # input_to_tpu, self.interpreter, 2, 1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_to_tpu)
        self.interpreter.invoke()
        # StatePartitionedCall:1
        mu = self.interpreter.get_tensor(self.output_details[1]['index'])
        # StatePartitionedCall:0
        logvar = self.interpreter.get_tensor(self.output_details[0]['index'])
        mu = numpy.squeeze(mu)
        logvar = numpy.squeeze(logvar)
        # Cast Interger outputs from TPU to float
        mu.astype(float)
        logvar.astype(float)

        # Change interpreter integer mu and var to floating domain
        mu = (mu - float(self.quant_zero_point_out1)) * self.quant_scale_out1
        logvar = (logvar - float(self.quant_zero_point_out2)) * self.quant_scale_out2

        # print(logvar)
        logvar = self.enc_head2logvar(logvar)

        d_kl = 0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)
        # print(f'NEW DKL:{d_kl}')
        return d_kl

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train or convert a BetaVAE model.')
    parser.add_argument(
        'action',
        choices=['train', 'convert', 'quantize'],
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
    elif args.action == 'quantize':
        # if args.quant_type == 'dynamic':
        #    full_model = torch.load(args.weights)
        #    full_model.to('cpu')
        #    q_model = torch.quantization.quantize_dynamic(
        #        full_model,
        #        {torch.nn.Linear},
        #        dtype=torch.qint8)
        #    torch.save(q_model, f'dq_{args.weights}')
        torch.backends.quantized.engine = 'qnnpack'
        model_sd = torch.load(args.weights, map_location=torch.device('cpu'))
        activation = {
            'relu': torch.nn.ReLU(),
            'leaky': torch.nn.LeakyReLU(0.1),
            'tanh': torch.nn.Tanh()}[args.activation]
        interpolation = {
            'nearest': PIL.Image.NEAREST,
            'bilinear': PIL.Image.BILINEAR,
            'bicubic': PIL.Image.BICUBIC}[args.interpolation]
        model_fp32 = QuantizedEncoder(
            n_latent=int(args.n_latent),
            n_chan=1 if args.grayscale else 3,
            input_d=tuple([int(i) for i in args.dimensions.split('x')]),
            activation=activation,
            head2logvar=args.head2learnwhat)
        model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        model_fp32.load_state_dict(model_sd)
        model_fp32.to('cpu')
        model_fp32.eval()
        model_fp32.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        model_fp32_fused = torch.quantization.fuse_modules(
            model_fp32,
            [
                ['enc_conv1', 'enc_conv1_bn', 'enc_conv1_af'],
                ['enc_conv2', 'enc_conv2_bn', 'enc_conv2_af'],
                ['enc_conv3', 'enc_conv3_bn', 'enc_conv3_af'],
                ['enc_conv4', 'enc_conv4_bn', 'enc_conv4_af']])
        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

        # Calibrate quantized model here
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(
                tuple([int(i) for i in args.dimensions.split('x')]),
                interpolation=interpolation)])
        if args.grayscale:
            transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(
                    tuple([int(i) for i in args.dimensions.split('x')]),
                    interpolation=interpolation),
                torchvision.transforms.Grayscale()])
        dataset = torchvision.datasets.ImageFolder(
            root=args.dataset,
            transform=transforms)
        train_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=64,
            shuffle=True,
            drop_last=True)

        for data in train_loader:
            input, _ = data
            mu, logvar = model_fp32(input)

        print('Calibration finished finished, saving weights...')
        model_int8 = torch.quantization.convert(model_fp32_prepared)
        torch.save(model_int8, f'sq_{args.weights}')
