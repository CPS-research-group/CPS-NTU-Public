import argparse
import os
import random
from glob import glob

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
# Original Droplets
# RAIN_KERNEL = numpy.array([
#     [0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.09, 0.09, 0.09, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.09, 0.09, 0.09, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.08, 0.08, 0.08, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.08, 0.08, 0.08, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.07, 0.07, 0.07, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.07, 0.07, 0.07, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.06, 0.06, 0.06, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.06, 0.06, 0.06, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.04, 0.04, 0.04, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.04, 0.04, 0.04, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.03, 0.03, 0.03, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.03, 0.03, 0.03, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.02, 0.02, 0.02, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.02, 0.02, 0.02, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0]])

#Bigger Droplets
# RAIN_KERNEL = numpy.array([
#     [0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0, 0],
#     [0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0],
#     [0, 0, 0, 0.09, 0.09, 0.09, 0.09, 0.09, 0, 0, 0],
#     [0, 0, 0, 0.09, 0.09, 0.09, 0.09, 0.09, 0, 0, 0],
#     [0, 0, 0, 0.08, 0.08, 0.08, 0.08, 0.08, 0, 0, 0],
#     [0, 0, 0, 0.08, 0.08, 0.08, 0.08, 0.08, 0, 0, 0],
#     [0, 0, 0, 0.07, 0.07, 0.07, 0.07, 0.07, 0, 0, 0],
#     [0, 0, 0, 0.07, 0.07, 0.07, 0.07, 0.07, 0, 0, 0],
#     [0, 0, 0, 0.06, 0.06, 0.06, 0.06, 0.06, 0, 0, 0],
#     [0, 0, 0, 0.06, 0.06, 0.06, 0.06, 0.06, 0, 0, 0],
#     [0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0],
#     [0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0],
#     [0, 0, 0, 0.04, 0.04, 0.04, 0.04, 0.04, 0, 0, 0],
#     [0, 0, 0, 0.04, 0.04, 0.04, 0.04, 0.04, 0, 0, 0],
#     [0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.03, 0, 0, 0],
#     [0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.03, 0, 0, 0],
#     [0, 0, 0, 0.02, 0.02, 0.02, 0.02, 0.02, 0, 0, 0],
#     [0, 0, 0, 0.02, 0.02, 0.02, 0.02, 0.02, 0, 0, 0],
#     [0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0],
#     [0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0],
#     [0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0],
#     [0, 0, 0, 0.01, 0.01, 0.01, 0.01, 0.01, 0, 0, 0]])

def rain_mask(size, intensity):
    dest = numpy.zeros((size[0], size[1]), dtype=numpy.uint8)
    noise = numpy.random.rand(size[0], size[1])
    dest[noise < intensity] = 255
    dest[noise >= intensity] = 0
    # Harsh droplets
    kernel = numpy.array([
        [0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.09, 0.09, 0.09, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.09, 0.09, 0.09, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.08, 0.08, 0.08, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.08, 0.08, 0.08, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.07, 0.07, 0.07, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.07, 0.07, 0.07, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.06, 0.06, 0.06, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.06, 0.06, 0.06, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.05, 0.05, 0.05, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.04, 0.04, 0.04, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.04, 0.04, 0.04, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.03, 0.03, 0.03, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.03, 0.03, 0.03, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.02, 0.02, 0.02, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.02, 0.02, 0.02, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0],
        [0, 0, 0, 0, 0.01, 0.01, 0.01, 0, 0, 0, 0]])
    dest = cv2.filter2D(dest, -1, kernel)
    return cv2.merge((dest, dest, dest))


def snow_mask(size, snow_level, flakes):
    RAIN_SPEED = 20
    # flakes = numpy.zeros((size[0], size[1]), dtype=numpy.uint8)
    noise = numpy.random.rand(RAIN_SPEED, size[1])
    new_flakes = numpy.zeros((RAIN_SPEED, size[1]), dtype=numpy.uint8)
    new_flakes[noise < snow_level] = 255
    new_flakes[noise >= snow_level] = 0
    flakes[RAIN_SPEED:, :] = flakes[:-RAIN_SPEED, :]
    flakes[:RAIN_SPEED, :] = new_flakes.astype(numpy.uint8)
    # snow_kernel = numpy.array([
    #     [0.1, 0.2, 0.3, 0.2, 0.1],
    #     [0.2, 0.3, 0.4, 0.3, 0.2],
    #     [0.3, 0.4, 0.5, 0.4, 0.3],
    #     [0.2, 0.3, 0.4, 0.3, 0.2],
    #     [0.1, 0.2, 0.3, 0.2, 0.1]])
    mask = cv2.filter2D(flakes, -1, 1.5 * RAIN_KERNEL)
    mask = cv2.merge((mask, mask, mask))
    return mask

def process_files(folder, rain_level):
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        mask = rain_mask(img.shape, rain_level)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
        mask[:,:,3] = 128
        img = cv2.add(img, mask)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(folder, file), img)
        
def process_snow_files(folder, snow_level, type_num):
    flakes = numpy.zeros((480, 640), dtype=numpy.uint8)
    imagelist = []
    imagelist = sorted(glob(folder + "/*.png"))
    frame_count = len(imagelist)
    if type_num == 3:     # 25% to 75%
        delay = int(frame_count * 0.25)
        stop = int(frame_count * 0.75)
        for i in range(delay, stop):
            img = cv2.imread(imagelist[i], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            mask = snow_mask(img.shape, snow_level, flakes)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
            mask[:,:,3] = 128
            img = cv2.add(img, mask)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(imagelist[i], img)
    elif type_num == 4:     # 50% onwards
        delay = int(frame_count * 0.5)
        stop = frame_count
        for i in range(delay, stop):
            img = cv2.imread(imagelist[i], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            mask = snow_mask(img.shape, snow_level, flakes)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
            mask[:,:,3] = 128
            img = cv2.add(img, mask)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(imagelist[i], img)
    else:
        for file in imagelist:
            img = cv2.imread(file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            mask = snow_mask(img.shape, snow_level, flakes)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
            mask[:,:,3] = 128
            img = cv2.add(img, mask)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(file, img)


def process_bw(folder, bw_level):
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = cv2.multiply(img[:, :, 2], (1 + bw_level))
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        cv2.imwrite(os.path.join(folder, file), img)


def process_video(path, rain_level, type):
    reader = cv2.VideoCapture(path)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(random.random() * frame_count)
    count = 0
    stop = 0
    if type == 0:
        stop = float('inf')
    elif type == 1:
        temp = int(random.random() * frame_count)
        stop = max(delay, temp)
        delay = min(delay, temp)
    elif type == 3:     # 25% to 75%
        delay = int(frame_count * 0.25)
        stop = int(frame_count * 0.75)
    elif type == 4:     # 50% onwards
        delay = int(frame_count * 0.5)
        stop = frame_count + 1
    else:
        delay = 0
        stop = frame_count + 1
    writer = cv2.VideoWriter(
        f'{path[:-4]}_rl{rain_level}_d{delay}_s{stop}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (640, 480))
    while reader.isOpened():
        count += 1
        ret, frame = reader.read()
        if not ret:
            break
        if count > delay and count < stop:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mask = rain_mask(frame.shape, rain_level)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGBA)
            mask[:,:,3] = 128
            frame = cv2.add(frame, mask)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        writer.write(frame)
    reader.release()
    writer.release()


def process_video_bw(path, bw_level, type):
    reader = cv2.VideoCapture(path)
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    delay = int(random.random() * frame_count)
    count = 0
    stop = 0
    if type == 0:
        stop = float('inf')
    elif type == 1:
        temp = int(random.random() * frame_count)
        stop = max(delay, temp)
        delay = min(delay, temp)
    elif type == 3:     # 25% to 75%
        delay = int(frame_count * 0.25)
        stop = int(frame_count * 0.75)
    elif type == 4:     # 50% onwards
        delay = int(frame_count * 0.5)
        stop = frame_count + 1
    else:
        delay = 0
        stop = frame_count + 1
    writer = cv2.VideoWriter(
        f'{path[:-4]}_bw{bw_level}_d{delay}_s{stop}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (640, 480))
    while reader.isOpened():
        count += 1
        ret, frame = reader.read()
        if not ret:
            break
        if count > delay and count < stop:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[:, :, 2] = cv2.multiply(frame[:, :, 2], (1 + bw_level))
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        writer.write(frame)
    reader.release()
    writer.release()


def process_video_snow(path, snow_level, type):
    reader = cv2.VideoCapture(path)
    flakes = numpy.zeros((480, 640), dtype=numpy.uint8)

    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        f'{path[:-4]}_sl{snow_level}.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (640, 480))
    count = 0
    while reader.isOpened():
        count += 1
        ret, frame = reader.read()
        if not ret:
            break
        RAIN_SPEED = 10
        noise = numpy.random.rand(RAIN_SPEED, 640)
        new_flakes = numpy.zeros((RAIN_SPEED, 640), dtype=numpy.uint8)
        new_flakes[noise < snow_level] = 255
        new_flakes[noise >= snow_level] = 0
        flakes[RAIN_SPEED:, :] = flakes[:-RAIN_SPEED, :]
        flakes[:RAIN_SPEED, :] = new_flakes.astype(numpy.uint8)
        # kernel = numpy.array([
        #     [0.1, 0.2, 0.3, 0.2, 0.1],
        #     [0.2, 0.3, 0.4, 0.3, 0.2],
        #     [0.3, 0.4, 0.5, 0.4, 0.3],
        #     [0.2, 0.3, 0.4, 0.3, 0.2],
        #     [0.1, 0.2, 0.3, 0.2, 0.1]])
        mask = cv2.filter2D(flakes, -1, 1.5 * RAIN_KERNEL)
        mask = cv2.merge((mask, mask, mask))
        #print(frame.shape)
        #print(mask.shape)
        frame = cv2.add(frame, mask)
        writer.write(frame)
    reader.release()
    writer.release()


def folder2vid(folder):
    dirname, vidname = os.path.split(os.path.normpath(folder))
    vidname = vidname + '.avi'
    writer = cv2.VideoWriter(
        os.path.join(dirname, vidname),
        cv2.VideoWriter_fourcc(*'XVID'),
        20.0,
        (640, 480))
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file), cv2.IMREAD_COLOR)
        writer.write(img)
    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Add OOD Rain to image/ video')
    parser.add_argument(
        '--input',
        help='Input video or folder of files')
    parser.add_argument(
        '--rain_level',
        default=0.0,
        help='Rain level to add')
    parser.add_argument(
        '--bw_level',
        help='Adjust brightness level')
    parser.add_argument(
        '--folder2vid',
        action='store_true',
        help='Don\'t add OOD, just convert folder to video')
    parser.add_argument(
        '--snow_level',
        default=0.0,
        help='Snow level')
    parser.add_argument(
        '--type',
        default=0,
        type=int,
        help='2 for whole video 1 level')
    args = parser.parse_args()
    if os.path.isdir(args.input):
        if args.folder2vid:
            folder2vid(args.input)
        elif args.rain_level != 0.0:
            process_files(args.input, float(args.rain_level))
        elif args.snow_level != 0.0:
            process_snow_files(args.input, float(args.snow_level), args.type)
        else:
            process_bw(args.input, float(args.bw_level))
    # if args.snow_level:
    #     process_video_snow(
    #         args.input,
    #         float(args.snow_level),
    #         type=0)
    else:
        if args.type == 0 and args.rain_level != 0.0:
            process_video(
                args.input,
                float(args.rain_level),
                type=0)
            process_video(
                args.input,
                float(args.rain_level),
                type=1)
        elif args.type == 0:
            process_video_bw(
                args.input,
                float(args.bw_level),
                type=0),
            process_video_bw(
                args.input,
                float(args.bw_level),
                type=1)
        elif args.rain_level != 0.0:
            process_video(
                args.input,
                float(args.rain_level),
                type=2)
        else:
            process_video_bw(
                args.input,
                float(args.bw_level),
                type=2)

