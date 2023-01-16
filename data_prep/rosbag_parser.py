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


def get_training_flows(args):
    """Get the training optical flows from a rosbag.

    Args:
        args: argparse object containing the user selected options.
    """
    output_dir_h = args.bagfile.replace('.bag', '_horiz')
    output_dir_v = args.bagfile.replace('.bag', '_vert')
    if os.path.exists(output_dir_h):
        shutil.rmtree(output_dir_h)
    if os.path.exists(output_dir_v):
        shutil.rmtree(output_dir_v)
    os.mkdir(output_dir_h)
    os.mkdir(output_dir_v)
    last_frame = None
    flow = None
    h_buf = [None, None, None, None, None, None]
    v_buf = [None, None, None, None, None, None]
    ptr = 0
    count = 0
    with rosbag.Bag(args.bagfile) as bag:
        topic_list = [f'/{args.host_name}/camera_node/image/compressed']
        for _, msg, t_stamp in bag.read_messages(topic_list):
            frame = cv2.imdecode(
                np.fromstring(msg.data, np.uint8),
                cv2.IMREAD_COLOR)
            frame = cv2.resize(frame, (160, 120))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    last_frame,
                    frame,
                    flow,
                    pyr_scale=0.5,
                    levels=1,
                    iterations=1,
                    winsize=15,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
                h_buf[ptr] = np.copy(flow[:, :, 0])
                v_buf[ptr] = np.copy(flow[:, :, 1])
                if count >= 5:
                    horiz = np.zeros((120, 160, 6))
                    vert = np.zeros((120, 160, 6))
                    for i in range(6):
                        horiz[:, :, i] = h_buf[ptr]
                        vert[:, :, i] = v_buf[ptr]
                        ptr = (ptr + 1) % 6
                    np.save(os.path.join(output_dir_h, f'{t_stamp}.npy'), horiz)
                    np.save(os.path.join(output_dir_v, f'{t_stamp}.npy'), vert)
                ptr = (ptr + 1) % 6
            last_frame = np.copy(frame)
            count += 1


def main():
    """Parse command line arguments and launch the corresponding function."""
    action_map = {
        'get_video': get_video,
        'measure_e2e_times': measure_e2e_times,
        'get_training_images': get_training_images,
        'get_ood_values': get_ood_values,
        'get_flows': get_training_flows,
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
