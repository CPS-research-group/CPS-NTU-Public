#!/usr/bin/env python3
"""Module to test that the required python packages have been installed on the
docker image used to run this ROS package.  Add this script to the default
launcher and verify there are no failures."""

# check python version
from platform import python_version
version = python_version()
if int(version.split('.')[0]) >2:
    print("Python {} \tOK".format(version))
else:
    print('Requires python version 3. {} is detected'.format(version))

# check numpy
try:
    import numpy as np
    version = np.__version__
    print("Numpy {} \tOK".format(version))
except ModuleNotFoundError as err:
    print("Numpy \tFAIL")
    print(err)

# check opencv version
import cv2
version = cv2.__version__
if int(version.split('.')[0]) >2:
    print("OpenCV {} \tOK".format(version))
    # check  opencv-contrib
    try:
        _=cv2.calcOpticalFlowFarneback(
            np.random.randint(255, size=(128,128)),
            np.random.randint(255, size=(128,128)),
            None,0.5,1,3,1,3,1.,0)
        print("Opencv-contrib \tOK")
    except ModuleNotFoundError as err:
        print("Opencv-contrib \tFAIL")
        print(err)
print('\n')

# check Pytorch
try:
    import torch
    from torch import nn
    from torchvision import transforms
    version = torch.__version__
    print("Pytorch {} \tOK".format(version))
except ModuleNotFoundError as err:
    print("Pytorch \tFAIL")
    print(err)

# for duckie only
# check PytorchLightening
print('\nDuckie')
try:
    import pytorch_lightning as pl
    version = python_version().split('.')
    if int(version[0])<3 or int(version[1]) < 6:
        print("PytorchLightening requires Python version 3.6 or above.")
    else:
        print("PytorchLightening \tOK")
except ModuleNotFoundError as err:
    print("PytorchLightening \tFAIL")
    print(err)

# check workstation GPU
print('\nWorkstation')
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU {} \taccessable".format(torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("GPU  \tnot accessable")
