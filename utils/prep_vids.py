#!/usr/bin/env python3


from typing import Tuple
import argparse
from glob import glob
import os
import random
import cv2
import numpy


# Finer droplets
RAIN_KERNEL = numpy.array([
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.05, 0.1, 0.05, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.045, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.045, 0.09, 0.09, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.04, 0.08, 0.04, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.035, 0.07, 0.035, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.03, 0.06, 0.03, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.025, 0.05, 0.025, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.015, 0.03, 0.015, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.01, 0.02, 0.01, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.005, 0.01, 0.005, 0, 0, 0, 0]])


def update_mask(mask: numpy.ndarray,
                rain_level: float,
                rain_speed: int) -> numpy.ndarray:
    """Generate a new rain mask given the previous one.
    
    Args:
        mask - previous rain mask image.
        rain_level - amount of rain at this time instance.
        rain_speed - how fast the rain should fall.
        
    Returns:
        The new rain mask.
    """
    noise = numpy.random.rand(rain_speed, mask.shape[1])
    new_drops = numpy.zeros((rain_speed, mask.shape[1]), dtype=numpy.uint8)
    new_drops[noise < rain_level] = 255
    new_drops[noise >= rain_level] = 0
    mask[rain_speed:, :] = mask[:-rain_speed, :]
    mask[:rain_speed, :] = new_drops
    return mask


def apply_mask(img: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """Given a rain or snow mask, apply it to the image.
    
    Args:
        img - image to add the mask to.
        mask - single channel mask of rain or snow.
        
    Returns:
        3-channel color image with the mask applied.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    mask = cv2.filter2D(mask, -1, 1.5 * RAIN_KERNEL)
    mask = cv2.merge((mask, mask, mask))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
    mask[:, :, 3] = 128
    img = cv2.add(img, mask)
    return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)


def adjust_brightness(img: numpy.ndarray, level: float) -> numpy.ndarray:
    """Adjust the brightness of an image.
    
    Args:
        img - input image.
        level - how much to change brightness (negative numbers are darker
            and positive numbers are brighter).
            
    Return:
        Image with adjusted brightness.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 2] = cv2.multiply(img[:, :, 2], (1 + level))
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def folder2vid(folder: str, output_d: Tuple[int]) -> None:
    """Create a video using all the pictures in a folder, where the frames
    appear in the order they are read from the folder (usually alphabetical).
    
    Args:
        folder - path to folder to convert.
        output_d - (width x height) desired dimensions of output video.
    """
    dirname, vidname = os.path.split(os.path.normpath(folder))
    vidname = vidname + '.avi'
    writer = cv2.VideoWriter(
        os.path.join(dirname, vidname),
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        output_d)
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_COLOR)
        img = cv2.resize(img, output_d)
        writer.write(img)
    writer.release()


def process_folder(folder: str,
                   rain_level: float,
                   rain_speed: int,
                   brightness_level: float,
                   mode: int,
                   output_d: Tuple[int]) -> None:
    """Add OOD with a given rain level and brightness level to every image
    file in a folder of images.

    Args:
        folder - path to folder of images.
        rain_level - amount of rain to add in (bigger = more).
        rain_speed - how fast the rain falls.
        bightness_level - change in image brightness (negative = darker,
            positive = lighter).
        mode - divide the files into 4 segments of equal length:
            1: [OOD, OOD, OOD, OOD]
            2: [ID,  ID,  OOD, OOD]
            3: [ID,  OOD, OOD, ID ]
        output_d - (width x height) desired dimensions of output images.
    """
    rain_mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    ood_start = 0
    ood_stop = frame_count
    if mode == 2:
        ood_start = int(frame_count * 0.5)
    elif mode == 3:
        ood_start = int(frame_count * 0.25)
        ood_stop = int(frame_count, * 0.75)
    count = 0
    for file in imagelist:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.resize(img, output_d)
        if count >= ood_start and count <= ood_stop:
            if rain_level > 0.0:
                rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                img = apply_mask(img, rain_mask)
            if brightness_level != 0.0:
                img = adjust_brightness(img, brightness_level)
        cv2.imwrite(file, img)
        count += 1


def process_video(path: str,
                  rain_level: float,
                  rain_speed: int,
                  brightness_level: float,
                  mode: int,
                  output_d: Tuple[int]) -> None:
    """Add OOD with a given rain level and brightness level to a video.

    Args:
        path - location of video.
        rain_level - amount of rain to add in (bigger = more).
        rain_speed - how fast the rain falls.
        bightness_level - change in image brightness (negative = darker,
            positive = lighter).
        mode - divide the video into 4 segments of equal length:
            1: [OOD, OOD, OOD, OOD]
            2: [ID,  ID,  OOD, OOD]
            3: [ID,  OOD, OOD, ID ]
        output_d - (width x height) desired dimensions of output video.
    """
    rain_mask = numpy.zeros(output_d[::-1], dtype=numpy.uint8)
    reader = cv2.VideoCapture(path)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        f'{path[:-4]}_rain{rain_level}_brightness{brightness_level}'
        f'_m{mode}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        output_d)
    ood_start = 0
    ood_stop = frame_count
    if mode == 2:
        ood_start = int(frame_count * 0.5)
    elif mode == 3:
        ood_start = int(frame_count * 0.25)
        ood_stop = int(frame_count * 0.75)
    count = 0
    while reader.isOpened():
        ret, frame = reader.read()
        if not ret:
            break
        frame = cv2.resize(frame, output_d)
        if count >= ood_start and count <= ood_stop:
            if rain_level > 0.0:
                rain_mask = update_mask(rain_mask, rain_level, rain_speed)
                frame = apply_mask(frame, rain_mask)
            if brightness_level != 0.0:
                frame = adjust_brightness(frame, brightness_level)
        writer.write(frame)
        count += 1
    reader.release()
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add OOD Rain to image/video.')
    parser.add_argument(
        '--input',
        help='Input video or folder of files. OOD will be added according to '
             'the "mode" argument.')
    parser.add_argument(
        '--rain_level',
        default=0.0,
        type=float,
        help='Rain level to add.')
    parser.add_argument(
        '--rain_speed',
        default=10,
        type=int,
        help='Speed of rainfall.')
    parser.add_argument(
        '--brightness_level',
        default=0.0,
        type=float,
        help='Adjust brightness level.')
    parser.add_argument(
        '--folder2vid',
        action='store_true',
        help='Don\'t add OOD, just convert folder to video.')
    parser.add_argument(
        '--output_width',
        default=640,
        type=int,
        help='Width of output images in pixels.')
    parser.add_argument(
        '--output_height',
        default=480,
        type=int,
        help='Height of output image in pixels.')
    parser.add_argument(
        '--mode',
        default=1,
        type=int,
        choices=[1, 2, 3],
        help='1: whole video or folder becomes OOD; 2: 1st half of video or '
             'folder remains ID, 2nd half becomes OOD; 3: 1st quarter of '
             'video or folder remains ID, quarters 2 and 3 become OOD, 4th '
             'quarter remains ID.')
    args = parser.parse_args()

    output_shape = (args.output_width, args.output_height)
    if os.path.isdir(args.input):
        if args.folder2vid:
            folder2vid(args.input, output_shape)
        else:
            process_folder(
                args.input,
                args.rain_level,
                args.rain_speed,
                args.brightness_level,
                args.mode,
                output_shape)
    else:
        process_video(
            args.input,
            args.rain_level,
            args.rain_speed,
            args.brightness_level,
            args.mode,
            output_shape)
