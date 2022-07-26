#!/usr/bin/env python3
# Module for Coral TPU, encoder only

import numpy as np

import torch
from torch import nn


class Encoder(nn.Module):

    def __init__(self, device, config, input_size):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.device = device
        self.group, self.nz, self.nd = 2, 12, 6
        self.mu1, self.mu2 = 0, 0
        self.var1 = torch.from_numpy(
            config["var_horizontal"] * np.ones(self.nz)).float()
        self.var2 = torch.from_numpy(
            config["var_vertical"] * np.ones(self.nz)).float()

        self.hiddenunits = 512

        # Input Size: 160,120 -> 152,113
        if self.input_size[0] == 113:
            self.grp1_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.nz)

            self.grp2_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=0,  bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.nz)

        # Input Size: 120,90 -> 116,87
        if self.input_size[0] == 87:
            self.grp1_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 1), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.nz)

            self.grp2_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 1), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.nz)

        # Input Size: 80,60 -> 77,56
        if self.input_size[0] == 56:
            self.grp1_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=(3, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(0, 1), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            # Kernel 2,2 works
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (2, 2), stride=(1, 1), padding=(0, 0), bias=False)
            # Kernel 3,3 is too large for (2,3 output)
            # self.grp1_conv4 = nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(1,0), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.nz)

            self.grp2_conv1 = nn.Conv2d(
                self.nd, 32, (5, 5), stride=(3, 3), padding=(3, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(0, 1), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            # Kernel 2,2 works
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (2, 2), stride=(1, 1), padding=(0, 0), bias=False)
            # Kernel 3,3 is too large for (2,3 output)
            # self.grp2_conv4 = nn.Conv2d(128, 256, (3,3), stride=(1,1), padding=(1,0), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.nz)

    def forward(self, x):
        # output_grp1 = torch.zeros([1, 1, 113, 152])
        # output_grp1[0, 0, :, :] = x[:, :, 0]
        # output_grp2 = torch.zeros([1, 1, 113, 152])
        # output_grp2[0, 0, :, :] = x[:, :, 1]

        output_grp1 = self.grp1_conv1(x)
        output_grp1 = self.grp1_conv1_bn(output_grp1)
        output_grp1 = self.grp1_conv1_ac(output_grp1)
        output_grp1 = self.grp1_conv2(output_grp1)
        output_grp1 = self.grp1_conv2_bn(output_grp1)
        output_grp1 = self.grp1_conv2_ac(output_grp1)
        output_grp1 = self.grp1_conv3(output_grp1)
        output_grp1 = self.grp1_conv3_bn(output_grp1)
        output_grp1 = self.grp1_conv3_ac(output_grp1)
        output_grp1 = self.grp1_conv4(output_grp1)
        output_grp1 = self.grp1_conv4_bn(output_grp1)
        output_grp1 = self.grp1_conv4_ac(output_grp1)
        output_grp1 = output_grp1.view(output_grp1.size(0), -1)
        output_grp1 = self.grp1_linear(output_grp1)

        output_grp2 = self.grp1_conv1(x)
        output_grp2 = self.grp2_conv1_bn(output_grp2)
        output_grp2 = self.grp2_conv1_ac(output_grp2)
        output_grp2 = self.grp2_conv2(output_grp2)
        output_grp2 = self.grp2_conv2_bn(output_grp2)
        output_grp2 = self.grp2_conv2_ac(output_grp2)
        output_grp2 = self.grp2_conv3(output_grp2)
        output_grp2 = self.grp2_conv3_bn(output_grp2)
        output_grp2 = self.grp2_conv3_ac(output_grp2)
        output_grp2 = self.grp2_conv4(output_grp2)
        output_grp2 = self.grp2_conv4_bn(output_grp2)
        output_grp2 = self.grp2_conv4_ac(output_grp2)
        output_grp2 = output_grp2.view(output_grp2.size(0), -1)
        output_grp2 = self.grp2_linear(output_grp2)

        return output_grp1, output_grp2

    def encode(self, input):
        # run on CPU/GPU
        # Input in NHWC format of shape (1,113, 152, 2)
        output_grp1, output_grp2 = self.forward(input)
        (mu_grp1, logvar_grp1) = output_grp1.chunk(2, 1)
        (mu_grp2, logvar_grp2) = output_grp2.chunk(2, 1)

        # CPU/GPU computation
        priorvar_grp1 = self.var1 * \
            torch.ones(logvar_grp1.size()).to(self.device)
        priorvar_grp2 = self.var2 * \
            torch.ones(logvar_grp2.size()).to(self.device)
        d_grp1 = ((mu_grp1 - self.mu1).pow(2) + logvar_grp1.exp() + priorvar_grp1 -
                  2. * torch.sqrt(logvar_grp1.exp() * priorvar_grp1)).sum(dim=1)

        d_grp2 = ((mu_grp2 - self.mu2) ** 2 + np.exp(logvar_grp2) + priorvar_grp2 -
                  2. * torch.sqrt(np.exp(logvar_grp2) * priorvar_grp2)).sum(dim=1)

        return (d_grp1, d_grp2)
