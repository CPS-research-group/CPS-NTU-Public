#!/usr/bin/env python3
"""Script to split dataset collected by the rosbag parser into training,
cross-validation, and test sets."""


import math
import os
import random
import shutil


if __name__ == '__main__':
    # Get a list of all flows, horizontal and veritcal
    files = []
    for folder in ['ID0_horiz', 'ID1_horiz']:
        for f in os.listdir(folder):
            files.append(os.path.join(folder, f))

    # Suffle the flow vectors and split 6/2/2 train/cv/test
    random.shuffle(files)
    train = files[:math.floor(0.6 * len(files))]
    val = files[math.floor(0.6 * len(files)):math.floor(0.8 * len(files))]
    cal = files[math.floor(0.8 * len(files)):]

    # Copy the files to their new respective locations
    for f in train:
        shutil.copy(f, os.path.join('Train', f.split('/')[1]))
    for f in val:
        shutil.copy(f, os.path.join('Validation', f.split('/')[1]))
    for f in cal:
        shutil.copy(f, os.path.join('Calibration', f.split('/')[1]))
