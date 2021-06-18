#!/usr/bin/env python3
"""module for ROS ood_node"""


import os
import sys
import numpy as np
import cv2
import torch
from torch import nn
import torchvision.transforms.functional as TF


class Encoder(nn.Module):
    """Encoder network of BVAE."""

    def __init__(self, device):
        super(Encoder, self).__init__()

        self.nc = 3
        self.nz = 30
        self.hidden_units = 256 * 2 * 3

        self.conv1 = nn.Conv2d(self.nc, 32, 5, stride=2, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_elu = nn.ELU()
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_elu = nn.ELU()
        self.conv3 = nn.Conv2d(64, 128, 5, stride=2, bias=False)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_elu = nn.ELU()
        self.conv4 = nn.Conv2d(128, 256, 5, stride=1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv4_elu = nn.ELU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.linear1 = nn.Linear(self.hidden_units, 1568)
        self.linear1_elu = nn.ELU()
        self.linear2 = nn.Linear(1568, 2*self.nz)


    def forward(self, x):

        x = torch.div(x, 255.)

        output = self.conv1(x)
        output = self.conv1_bn(output)
        output = self.conv1_elu(output)

        output = self.conv2(output)
        output = self.conv2_bn(output)
        output = self.conv2_elu(output)

        output = self.conv3(output)
        output = self.conv3_bn(output)
        output = self.conv3_elu(output)

        output = self.conv4(output)
        output = self.conv4_bn(output)
        output = self.conv4_elu(output)

        output = self.maxpool1(output)
        output = self.maxpool2(output)

        output = output.view(output.size(0), -1)
        output = self.linear1(output)
        output = self.linear1_elu(output)

        output = self.linear2(output.view(output.size()[:2]))

        return output.chunk(2, 1)

    def encode(self, input):
        mu, logvar = self.forward(input)
        d_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        return d_kl


class Detector(nn.Module):
    """The BVAE OOD Detector."""

    def __init__(self, modelfile, device):
        super(Detector, self).__init__()

        self.new_image_size = (117,149)

        self.config = {"topz":[]}
        for (key, value) in np.loadtxt(modelfile + "_config",
                delimiter=":", comments="#", dtype=str) :
            if key == "topz":
                self.config[key].append(int(value))
            else:
                self.config[key] = float(value)

        self.device = device
        self.encoder = Encoder(self.device)
        self.load_state_dict(torch.load(modelfile, map_location = device)["model_state_dict"])
        self.eval()

        self.topz = self.config["topz"]
        self.threshold = self.config["threshold"]

    def check(self, data):

        data = cv2.resize(data, (self.new_image_size[1],self.new_image_size[0]) )
        data = TF.to_tensor(data)

        with torch.no_grad():
            d_kl = self.encoder.encode(data.unsqueeze(0).to(self.device)).numpy()[0]
            d_kl = np.sum(d_kl[i] for i in self.topz)
            predict = True
            if d_kl < self.threshold:
                predict = False

        return {"isood":predict, "value":d_kl}


###########################################
# Test outside ROS node
###########################################
if __name__ == "__main__":
    import time
    from glob import glob

    modelfile = input("Enter model file : < betav_nuscenes.pt | betav_synthia.pt > : ").strip(" ")
    if "/" not in modelfile: modelfile = "./" + modelfile
    if os.path.isfile(modelfile) is False:
        print("Can not access ", modelfile)
        sys.exit()

    configfile = modelfile + "_config"
    if os.path.isfile(configfile) is False:
        print("Please place {} file in the same folder of modelfile.".format(configfile))
        sys.exit()

    imagesequence_folder = input("Enter the folder stores test images : ").strip(" ")
    if "/" not in imagesequence_folder: imagesequence_folder = "./" + imagesequence_folder
    if os.path.isdir(imagesequence_folder) is False:
        print("Can not access ", imagesequence_folder)
        sys.exit()

    ## check computing device
    tpu = input("Use TPU? < no | yes > : ")
    if tpu.lower()=="yes":
        print("TPU is not supported now.")
        sys.exit()
    else:
        device = torch.device("cpu")
        tpu_ready = True
        print("Use CPU")


    ## initialize detector
    ts = time.time()
    detector = Detector(modelfile, device)
    print("{:12.3f}\tDetector is initialized.".format(time.time()-ts))

    ## run test
    for imagefile in sorted(glob(imagesequence_folder + "/*")):
        ts = time.time()
        ret = detector.check(cv2.imread(imagefile))
        print("{:12.3f}\t{}  {:.6f})".\
                format(time.time()-ts, ret["isood"], ret["value"]))
