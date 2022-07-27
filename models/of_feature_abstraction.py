#!/usr/bin/env python3
from cv2 import DESCRIPTOR_MATCHER_BRUTEFORCE_L1
import numpy as np
from glob import glob
import cv2
import h5py
import os
import math
'''
	This script is a demo of feature abstraction using optical flow operations.

	Place all scene folders in one folder, e.g. data/nuscenes-v1.0-mini/ 

	Each scene will be splitted into a test (1-48 frames) and a train (49-last frames) feature file in hdf5 format
    
    Each image is sharpened for the top one-third of the image before feature abstraction happens.
    

'''


class FeatureAbstraction:
    def __init__(self, sourcepath, dimensions=120, interpolation=3):

        # Specify output root
        # self.dstroot = sourcepath.split("/")[0] + "/"
        self.sourcepath = sourcepath
        self.interpolation = interpolation
        self.dim1 = dimensions
        self.dim2 = math.ceil(self.dim1 * 1.333)
        current_dir = os.getcwd()
        new_dir_name = f"{self.dim1}x{self.dim2}"
        self.new_dir = os.path.join(
            current_dir, "OF", str(interpolation), new_dir_name)
        

    def extract(self):
        if (os.path.isdir(self.new_dir)):
            return self.new_dir
        else:
            os.makedirs(self.new_dir)
        self.dstroot = self.new_dir
        # List file for train and test loaders
        self.list_train, self.list_test = [], []

        self.newdim = (int(self.dim2*2), int(self.dim1*2))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        for scenefolder in sorted(glob(self.sourcepath + "*")):
            frames = []
            # It is time series. Frame order matters !!!
            for imagefile in sorted(glob(scenefolder + "/*.png")):
                frames.append(imagefile)

            # Fetch the 1st frame
            im1 = cv2.cvtColor(cv2.resize(cv2.imread(
                frames[0]), self.newdim, interpolation=self.interpolation), cv2.COLOR_BGR2GRAY)
            im1 = clahe.apply(im1)
            # Sharpen
            kernel = np.array([[0, -1, 0],      # Sharpen kernel from wiki
                               [-1, 5, -1],
                               [0, -1, 0]])

            # 1 Third Top Sharp
            image_sharp = cv2.filter2D(
                src=im1[:int(self.newdim[1]/3), :], ddepth=-1, kernel=kernel)
            im1[:int(self.newdim[1]/3), :] = image_sharp

            features_x, features_y = [], []
            im2, flow = None, None
            for i in range(1, len(frames)):
                if im2 is None:
                    im2 = cv2.cvtColor(cv2.resize(cv2.imread(
                        frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)
                    im2 = clahe.apply(im2)

                    # 1 Third Top Sharp
                    image_sharp = cv2.filter2D(
                        src=im2[:int(self.newdim[1]/3), :], ddepth=-1, kernel=kernel)
                    im2[:int(self.newdim[1]/3), :] = image_sharp

                else:
                    im1 = im2[:]
                    im2 = cv2.cvtColor(cv2.resize(cv2.imread(
                        frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)
                    im2 = clahe.apply(im2)

                    # 1 Third Top Sharp
                    image_sharp = cv2.filter2D(
                        src=im2[:int(self.newdim[1]/3), :], ddepth=-1, kernel=kernel)
                    im2[:int(self.newdim[1]/3), :] = image_sharp

                flow = cv2.calcOpticalFlowFarneback(im1, im2, flow,
                                                    pyr_scale=0.5, levels=1, iterations=1,
                                                    winsize=15, poly_n=5, poly_sigma=1.1,
                                                    flags=0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
                # For original dimensions of 320, 240
                features_x.append(cv2.resize(
                    flow[..., 0], None, fx=0.5, fy=0.5))
                features_y.append(cv2.resize(
                    flow[..., 1], None, fx=0.5, fy=0.5))
                # Dimensions are already of of 160, 120)
                # features_x.append(flow[..., 0])
                # features_y.append(flow[..., 1])

            # Write optic flow fields of one video episode to a h5 file
            # First 48 frames for test, rest for train
            trainfile = self.dstroot + "/train." + \
                scenefolder.split("/")[-1] + ".h5"
            with h5py.File(trainfile, "w") as f:
                f.create_dataset("x", data=features_x[48:])
                f.create_dataset("y", data=features_y[48:])
            self.list_train.append(trainfile)

            testfile = self.dstroot + "/test." + \
                scenefolder.split("/")[-1] + ".h5"
            with h5py.File(testfile, "w") as f:
                f.create_dataset("x", data=features_x[:48])
                f.create_dataset("y", data=features_y[:48])
            self.list_test.append(testfile)

        h5fillist_train = self.dstroot + "/" + \
            self.sourcepath.split("/")[-2]+".train"
        with open(h5fillist_train, "w") as f:
            for scene in self.list_train:
                f.write(scene+"\n")

        h5filelist_test = self.dstroot + "/" + \
            self.sourcepath.split("/")[-2]+".test"
        with open(h5filelist_test, "w") as f:
            for scene in self.list_test:
                f.write(scene+"\n")

        print("See feature extraction results in {} and {}".format(
            h5fillist_train, h5filelist_test))

        return self.new_dir


if __name__ == "__main__":
    sourceroot = 'data_train/'
    if os.path.isdir(sourceroot):
        # dimensions = (320, 240)  # -> (160, 120)
        # dimensions = (256, 192)  # -> (128, 96)
        # dimensions = (192, 144)  # -> (96, 72)
        # dimensions = (128, 96)  # -> (64, 48)
        # dimensions = (64, 48)  # -> (32, 24)
        FeatureAbstraction(sourceroot)
