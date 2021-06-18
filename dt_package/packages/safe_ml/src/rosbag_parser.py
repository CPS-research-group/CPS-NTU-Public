#!/usr/bin/env python3
"""Parse Rosbags that were recorded during experiments or testing."""


import argparse
import os
import shutil
import zipfile


import cv2
import rosbag
import numpy as np
import torch
import yaml


from skimage.metrics import structural_similarity
from ood_module_betav import Detector


def get_video(args):
    """Write the sequence of frames from the duckiebot camera to a video.

    Args:
        args: argparse object containing the user selected options.
    """
    with rosbag.Bag(args.bagfile) as bag:
        writer = cv2.VideoWriter(
            f'{args.bagfile.replace(".bag", "")}.avi',
            cv2.VideoWriter_fourcc(*'MJPG'),
            30.0,
            (480, 640))
        topics = [f'/{args.host_name}/camera_node/image/compressed']
        for _, msg, _ in bag.read_messages(topics):
            frame = cv2.imdecode(
                np.fromstring(msg.data, np.uint8),
                cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (640, 480))
            writer.write(frame)
        writer.release()


def measure_e2e_times(args):
    """Write the frames captured by the camera into a folder with a line at
    30cm and 60cm away from the duckiebot.  This can be used to conviniently
    determine at what time stamp an object entered the duckiebot's risk zone.
    Also stamp images with an indication if there was motion or not to check
    when the approaching object stopped.

    Args:
        args: argparse object containing the user selected options.
    """
    with rosbag.Bag(args.bagfile) as bag:
        output_dir = args.bagfile.replace('.bag', '')
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.mkdir(output_dir)
        last_frame = np.zeros((120, 160))
        topics = [f'/{args.host_name}/camera_node/image/compressed']
        for _, msg, t_stamp in bag.read_messages(topics):
            frame = cv2.imdecode(
                np.fromstring(msg.data, np.uint8),
                cv2.IMREAD_COLOR)
            # The pixel rows corresponding to distance from front of Duckiebot
            # were determined experimentally
            alpha_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.line(alpha_image, (0, 241), (640,241), (0, 255, 0), thickness=1)
            cv2.line(alpha_image, (0, 200), (640,200), (255, 0, 0), thickness=1)
            frame = cv2.addWeighted(frame, 1.0, alpha_image, 0.8, 0.0)
            cv2.putText(
                img=frame,
                text=f'{t_stamp}',
                org=(10, 450),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2)
            resized_frame = cv2.resize(frame, (160, 120))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
            ssim = structural_similarity(
                resized_frame,
                last_frame,
                data_range=last_frame.max() - last_frame.min())
            last_frame = resized_frame
            cv2.putText(
                img=frame,
                text=f'SSIM: {ssim}',
                org=(10,30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=2)
            cv2.putText(
                img=frame,
                text=f'{"STOPPED" if ssim > 0.98 else ""}',
                org=(10,60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 255),
                thickness=2)
            cv2.imwrite(os.path.join(output_dir, f'{t_stamp}.png'), frame)


def get_training_images(args):
    """Write the raw images from the Rosbag to a zipfile for use in model
    training.

    Args:
        args: argparse object containing the user selected options.
    """
    output_dir = args.bagfile.replace('.bag', '')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    with rosbag.Bag(args.bagfile) as bag:
        topic_list = [f'/{args.host_name}/camera_node/image/compressed']
        for _, msg, t_stamp in bag.read_messages(topic_list):
            frame = cv2.imdecode(
                np.fromstring(msg.data, np.uint8),
                cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(output_dir, f'{t_stamp}.png'), frame)
    with zipfile.ZipFile(args.bagfile.replace('.bag', '.zip'), 'w') as zipf:
        for base, _, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(base, file))


def get_ood_values(args):
    """Iterate through images in the rosbag and find the images with an OOD
    score above some threshold.

    Args:
        args: argparse object containing the user selected options.
    """
    params_file = os.path.join(os.path.dirname(__file__), 'ood_params.yml')
    params = {}
    with open(params_file, 'r') as paramf:
        params = yaml.safe_load(paramf.read())
    modelfile = os.path.join(os.path.dirname(__file__), params['model'])
    torchdevice = params["torch_device"]
    model = Detector(modelfile, torch.device(torchdevice))
    with rosbag.Bag(args.bagfile) as bag:
        topic_list = [f'/{args.host_name}/camera_node/image/compressed']
        for _, msg, t_stamp in bag.read_messages(topic_list):
            frame = cv2.imdecode(
                np.fromstring(msg.data, np.uint8),
                cv2.IMREAD_COLOR)
            result = model.check(frame)
            if result['value'] > args.threshold:
                cv2.imwrite(f'{t_stamp}.png', frame)
                print(f'Image {t_stamp}.png has OOD score {result["value"]}')


def main():
    """Parse command line arguments and launch the corresponding function."""
    action_map = {
        'get_video': get_video,
        'measure_e2e_times': measure_e2e_times,
        'get_training_images': get_training_images,
        'get_ood_values': get_ood_values,
    }
    parser = argparse.ArgumentParser(
        'Parse a rosbag and get useful statistics / visualizations on SafeML '
        'performance.')
    parser.add_argument(
        'action',
        help='Action to perform on the rosbag.',
        choices=list(action_map.keys()))
    parser.add_argument(
        '-H',
        '--host_name',
        default='safeducke1',
        help='Name of host where the ROS master is running.')
    parser.add_argument(
        '-t',
        '--threshold',
        type=float,
        default=0.11,
        help='Threshold for use with "get_ood_values" action')
    parser.add_argument(
        'bagfile',
        help='Bagfile to parse.')
    args = parser.parse_args()
    action_map[args.action](args)


if __name__ == '__main__':
    main()
