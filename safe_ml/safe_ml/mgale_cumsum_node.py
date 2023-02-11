"""Monothlithic OOD Detection Node.  Perform all preprocessing, network
inferences, ICP, and Martingale calcutions in a single ROS2 node."""

import argparse
import io
import json
import os
import pkgutil
import sys
import time
import tracemalloc


import cv2
import numpy
import rclpy
import torch


from .bvae import Encoder
from rclpy.node import Node
from rclpy.qos import (
     QoSProfile,
     QoSDurabilityPolicy,
     QoSHistoryPolicy,
     QoSReliabilityPolicy,
     QoSPresetProfiles)
from std_msgs.msg import Bool, Float64MultiArray, Header
from sensor_msgs.msg import CompressedImage
from torch import nn
from torchvision.transforms import functional
from ament_index_python.packages import get_package_share_directory


class MartingaleCumSumNode(Node):
    """Monolithic node to detect OOD images.  All preprocessing, inferences,
    ICP, and Martingale calculations are performed here.
    
    Args:
        weights - weights file to use for the encoder.
        alpha_cal - alpha calibration set file to use for ICP.
        device - currently on 'CUDA' or 'CPU' supported.
        window - length of Martingale window.
        decay - decay constant for cummulative summation.
    """

    def __init__(self,
                 weights: str,
                 alpha_cal: str,
                 device: str,
                 window: int,
                 partition: str,
                 name: str) -> None:
        # ROS2 node initialization
        super().__init__(name)
        self.get_logger().info('Preparing Martingale/CUMSUM node...')

        # Load calibration data
        self.get_logger().info('Loading calibration data...')
        cal_data = {}
        cal_path = os.path.join(
            get_package_share_directory('safe_ml'),
            'bvae_model',
            alpha_cal)
        with open(cal_path, 'r') as cal_f:
            cal_data = json.loads(cal_f.read())
        self.n_latent = cal_data['PARAMS']['n_latent']
        self.n_chan = cal_data['PARAMS']['n_chan']
        self.input_d = tuple(cal_data['PARAMS']['input_d'])
        #decay = cal_data['PARAMS']['decay']
        #PATCH
        decay = 1.0
        del cal_data['PARAMS']
        #self.alpha_cal = numpy.zeros((1, len(list(cal_data[partition]['dkls']))))
        self.z_mask = numpy.zeros(self.n_latent)
        self.z_mask[cal_data[partition]['top_z'][0][0]] = 1
        self.alpha_cal = numpy.sum(
            numpy.array(cal_data[partition]['dkls']) * self.z_mask,
            axis=1)

        # Setup Martingale
        self.get_logger().info('Setting up Martingale...')
        self.ptr = 0
        self.window = window
        self.past_dkls = numpy.zeros(self.window)
        self.eps = numpy.linspace(0, 1, 100)

        # Setup cummalitve sum
        self.decay = decay
        self.csum = 0

        # Setup logging
        self.get_logger().info('Setting up logfiles...')
        self.results = open(f'results_mgale_cumsum_{partition}.csv', 'w')
        self.results.write('start,martingale,end,m,cumsum\n')

        # Establish subscribers
        self.get_logger().info('Creating subscribers...')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)
        dkl_sub = self.create_subscription(
            Float64MultiArray,
            '/ood_detector/dkls',
            self.run_detect,
            qos_profile)
        shutdown_sub = self.create_subscription(
            Bool,
            '/mock_camera/done',
            self.stop,
            qos_profile)
        self.get_logger().info('martingale and cumsum node setup complete!')

    def run_detect(self, msg: Float64MultiArray) -> None:
        """Perform OOD detection on an incoming frame.

        Args:
            msg - CompressedImage message from camera.
        """
        start_time = time.time()
        self.icp(numpy.array(msg.data))
        m = self.martingale()
        martingale_time = time.time()
        self.cumsum(m)
        end_time = time.time()
        self.results.write(
            f'{start_time},{martingale_time},{end_time},'
            f'{m},{self.csum}\n')

    def icp(self, dkls: numpy.ndarray) -> numpy.ndarray:
        """Find the percentage of samples in the calibration set with a higher
        KL-divergence than the input sample (Inductive Conformal Prediction).

        Args:
            dkls - a (1 x n_latent) vector of KL-divergence for each latent
                dimension.

        Returns:
            A (n_partition x 1) vector where entry p_i is the percentage of
            samples in the calibration set with a higher KL-divergence than
            the input for the i-th data partition.
        """
        dkl = numpy.sum(dkls * self.z_mask)
        p_i = max(numpy.count_nonzero(self.alpha_cal > dkl), 2) / self.alpha_cal.size
        self.past_dkls[self.ptr] = p_i
        self.ptr = (self.ptr + 1) % self.window
        return p_i

    def martingale(self) -> numpy.ndarray:
        """Compute the martingale on the detector's ICP output window.

        Returns:
            A (1 x n_partition) vector of martingale values for all samples
            currently stored in this class's ICP output window.
        """
        m = 0
        for i in range(100):
            m += numpy.product(self.eps[i] * numpy.power(self.past_dkls, self.eps[i] - 1))
        return m

    def cumsum(self, m: float) -> None:
        """Given a new sample, update this class's cumulative summation, which
        is used as the current OOD score.  The cumulative summation can be
        accessed at any time to check the status of a frame as OOD.

        Args:
            m - the Martingale resulting from an input frame.
        """
        m = numpy.nan_to_num(m)
        self.csum = numpy.maximum(0, self.csum + numpy.log(m) - self.decay)

    def stop(self, msg: Bool) -> None:
        """Stop this node."""
        self.get_logger().warn('Received message to STOP!')
        self.results.flush()
        self.results.close()
        time.sleep(15)
        self.executor.shutdown(1)
        raise Exception('Received STOP message.')


def main():
    tracemalloc.start()
    parser = argparse.ArgumentParser('OOD Detector Node')
    parser.add_argument(
        '--weights',
        help='weights file.')
    parser.add_argument(
        '--alpha_cal',
        help='calibration file.')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'tpu', 'tflite'],
        default='cuda',
        help='torch device.')
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        help='Martingale window size.')
    parser.add_argument(
        '--partition',
        help='Partition to use.')
    args, _ = parser.parse_args()
    rclpy.init(args=sys.argv)
    ood_detector_node = MartingaleCumSumNode(
        args.weights,
        args.alpha_cal,
        args.device,
        args.window,
        args.partition,
        'mgale_cumsum_node')
    try:
        rclpy.spin(ood_detector_node)
    except KeyboardInterrupt:
        print('Martingale / CUMSUM Node Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Martingale / CUMSUM Node Received Exception: {str(error)}')
    finally:
        with open('cpu_stats_mgale.txt') as fp:
            fp.write(str(tracemalloc.get_traced_memory()))
        with open('cuda_stats.txt', 'w') as fp:
            fp.write(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        ood_detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
