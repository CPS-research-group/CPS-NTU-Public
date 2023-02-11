import argparse
import os
import re
import pkgutil
import sys
import time


import cv2
import numpy as np
import torch
import rclpy


from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSPresetProfiles)
from std_msgs.msg import Bool, Float64MultiArray, Header
from sensor_msgs.msg import CompressedImage

try:
    from pycoral.utils import edgetpu
except:
    pass
# from pycoral.utils import dataset
# from pycoral.adapters import common

from ament_index_python.packages import get_package_share_directory

from .bi3dof import Detector


class OFDetectorEncNode(Node):
    """Detect OF OOD images.
    All preprocessing, inferences,
    ICP, and Martingale calculations are performed here.

    Args:
        weights - weights file to use for the encoder.
        device - currently on 'CUDA' or 'CPU' supported.
    """

    def __init__(self, args):
        super().__init__(f'of_detector_enc{args.group}_node')
        self.get_logger().info('Preparing opticalflow detector encoder node...')
        self.get_logger().info('Loading model on device...')
        # Device can be cpu, cuda or tpu
        if args.device == 'tpu' or args.device is None:
            args.device = None
        elif args.device == "tflite":
            args.device = "TFLITE"
        else:
            args.device = torch.device(args.device)

        if 'bi3dof' in args.weights:
            self.config = {'metric': 0, 'var_horizontal': 1, 'var_vertical': 1}
            # Initialze cene specific parameters
            self.counter = 0
            self.scene = args.video
            self.scene_length = int(self.scene.split('_')[2])
            self.ood_start = -1
            self.ood_end = -1
            match = re.search('.*_m(\d).*', self.scene)
            if match:
                match_val = int(match.groups()[0])
                if match_val == 1:
                    self.ood_start = 0
                    self.ood_end = self.scene_length
                elif match_val == 2:
                    self.ood_start = int(0.5 * self.scene_length)
                    self.ood_end = self.scene_length
                elif match_val == 3:
                    self.ood_start = int(0.25 * self.scene_length)
                    self.ood_end = int(0.75 * self.scene_length)
            # Initialze model specific parameters
            self.image_size = [int(i) for i in args.weights.split(
                '.')[-2].split('_')[1].split('x')]
            self.cropped_size = [int(i) for i in args.weights.split(
                '.')[-2].split('_')[2].split('x')]
            args.crop_height = self.cropped_size[0]
            args.crop_width = self.cropped_size[1]
            args.n_frames = int(args.weights.split('.')[-2].split('_')[5])
            self.crop_width1 = int(
                (self.image_size[1] - self.cropped_size[1]) / 2)
            self.crop_height1 = int(
                (self.image_size[0] - self.cropped_size[0]) / 2)
            if ((self.image_size[1] - self.cropped_size[1])) % 2 != 0:
                self.crop_width2 = self.crop_width1 + 1
            else:
                self.crop_width2 = self.crop_width1
            if ((self.image_size[0] - self.cropped_size[0])) % 2 != 0:
                self.crop_height2 = self.crop_height1 + 1
            else:
                self.crop_height2 = self.crop_height1
            self.previous_frame = None
            self.flow = None
            self.of_idx = 0
            self.of_depth = int(args.weights.split('.')[-2].split('_')[5])
            self.interpolation = int(
                args.weights.split('.')[-2].split('_')[3][-1])
            self.of_frames = np.zeros(
                (2, self.cropped_size[0], self.cropped_size[1], self.of_depth))
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            model_file = os.path.join(
                get_package_share_directory('safe_ml'),
                'of_model',
                args.weights
            )
            print(model_file)

            self.detector = Detector(
                model_file, args)
            self.group = int(args.group)

            if args.device is None:
                # TPU
                self.result_file = "score_bi3dof_enc_tpu_{}.csv".format(
                    os.uname().machine)
            elif args.device == "TFLITE":
                self.result_file = "score_bi3dof_enc{}_tflite_{}.csv".format(args.group, 
                    os.uname().machine)
            else:
                # CPU
                self.result_file = "score_bi3dof_enc{}_cpu_{}.csv".format(args.group,
                    os.uname().machine)

            self.get_logger().info('Establishing logfile...')
            self.of_detection_times_f = open(self.result_file, "w")
            self.of_detection_times_f.write(
                "start,inference,end,\
                value_{self.group},ground_truth\n")

            self.get_logger().info('Creating subscribers...')
            qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
                history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
                durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)
            if self.group == 0:
                self.flow_sub = self.create_subscription(
                    Float64MultiArray,
                    '/ood_detector/hflow',
                    self.detect_of,
                    qos_profile)
            else:
                self.flow_sub = self.create_subscription(
                    Float64MultiArray,
                    '/ood_detector/vflow',
                    self.detect_of,
                    qos_profile)
            self.get_logger().info('ood detector encoder node setup complete.')
            shutdown_sub = self.create_subscription(
                Bool,
                '/mock_camera/done',
                self.stop,
                qos_profile)

    def detect_of(self, msg):
        ground_truth = 0
        if self.counter >= self.ood_start and self.counter <= self.ood_end:
            ground_truth = 1
        start_time = time.time()

        data = np.array(msg.data).reshape(
            1,
            self.cropped_size[0],
            self.cropped_size[1],
            self.of_depth)
        ret = self.detector.check_group(data, self.group)
        inference_time = time.time()
        end_time = inference_time
        self.of_detection_times_f.write(
            "{},{},{},{},{}\n".format(start_time, inference_time, end_time,
            ret["value"], ground_truth))
        self.counter += 1

    def stop(self, msg):
        self.get_logger().warn('Received message to STOP!')
        self.of_detection_times_f.flush()
        self.of_detection_times_f.close()
        raise Exception('Received STOP message.')

def main():
    parser = argparse.ArgumentParser('OOD OF Encoder Node')
    parser.add_argument(
        '--weights',
        help='weights file.')
    parser.add_argument(
        '--video',
        help='video file input.')
    parser.add_argument(
        '-t',
        '--type',
        help='variation to run')
    parser.add_argument(
        '-d',
        '--device', type=str, default='cpu',
        help='cpu or cuda or tpu')
    parser.add_argument(
        '-m',
        '--model',
        help='model file')
    parser.add_argument(
        '-s',
        '--scene',
        help='scene name')
    parser.add_argument(
        '--group',
        help='horizontal (0), vertical (1)',
        choices=[0, 1],
        type=int)
    args = parser.parse_args()
    if args.device:
        if args.device == 'tpu':
            device = None
        else:
            device = args.device
    rclpy.init(args=sys.argv)
    ood_detector_enc_node = OFDetectorEncNode(args)
    try:
        rclpy.spin(ood_detector_enc_node)
    except KeyboardInterrupt:
        print('Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Received Exception: {str(error)}')
    finally:
        if device == 'cuda':
            with open('cuda_stats.txt', 'w') as fp:
                fp.write(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
        if "bi3dof" in args.weights:
            ood_detector_enc_node.of_detection_times_f.close()
        ood_detector_enc_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
