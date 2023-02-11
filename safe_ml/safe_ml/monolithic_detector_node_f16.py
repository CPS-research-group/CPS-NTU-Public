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
import tflite_runtime.interpreter as tflite


import cv2
import numpy
import rclpy
import torch
import torchvision


from .bvae import Encoder, Encoder_TPU, Encoder_TFLITE_BVAE
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSPresetProfiles)
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
except:
    pass


class MonolithicDetectorNode(Node):
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
                 window: int) -> None:
        # ROS2 node initialization
        super().__init__('monolithic_detector_node')
        self.get_logger().info('Preparing monolithic detector node...')
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
        self.interpolation = cal_data['PARAMS']['interpolation']
        # decay = {}
        # decay['rain'] = cal_data['PARAMS']['decay']['rain']
        # decay['brightness'] = cal_data['PARAMS']['decay']['brightness']
        # Patch
        decay = {'rain': 1.0, 'brightnewss': 1.0}
        del cal_data['PARAMS']
        self.alpha_cal = numpy.zeros((
            len(cal_data.keys()),
            len(list(cal_data.values())[0]['dkls'])))
        self.z_mask = numpy.zeros((len(cal_data.keys()), self.n_latent))
        self.partitions = []
        for idx, item in enumerate(cal_data.items()):
            self.partitions.append(item[0])
            self.z_mask[idx, item[1]['top_z'][0][0]] = 1
            self.alpha_cal[idx, :] = numpy.sum(
                numpy.array(item[1]['dkls']) * self.z_mask[idx, :],
                axis=1)

        # Setup encoder network
        self.get_logger().info('Loading trained weights...')
        # Device can be cpu, cuda or tpu
        if device == 'tpu' or device is None:
            self.device = None
            self.interpreter = edgetpu.make_interpreter(
                os.path.join(
                    get_package_share_directory('safe_ml'),
                    'bvae_model',
                    weights))
            self.interpreter.allocate_tensors()
            self.encoder = Encoder_TPU(
                self.interpreter,
                n_latent=self.n_latent,
                n_chan=self.n_chan,
                input_d=self.input_d,
                interpolation=self.interpolation,
                head2logvar='var')
        elif device == 'tflite':
            self.device = "TFLITE"
            self.interpreter = tflite.Interpreter(
                os.path.join(
                    get_package_share_directory('safe_ml'),
                    'bvae_model',
                    weights))
            self.interpreter.allocate_tensors()
            self.encoder = Encoder_TFLITE_BVAE(
                self.interpreter,
                n_latent=self.n_latent,
                n_chan=self.n_chan,
                input_d=self.input_d,
                interpolation=self.interpolation,
                head2logvar='var')
        else:
            self.device = torch.device(device)
            self.encoder = Encoder(
                n_latent=self.n_latent,
                n_chan=self.n_chan,
                input_d=self.input_d,
                head2logvar='var')
            self.encoder.load_state_dict(torch.load(
                os.path.join(
                    get_package_share_directory('safe_ml'),
                    'bvae_model',
                    weights), map_location=self.device))
            self.encoder = self.encoder.half()
            self.encoder.eval()
            self.encoder.to(self.device)

        # Setup Martingale
        self.get_logger().info('Setting up Martingale...')
        self.ptr = 0
        self.window = window
        self.past_dkls = numpy.zeros((len(self.partitions), self.window))
        self.eps = numpy.linspace(0, 1, 100)

        # Setup cummalitve sum
        print(f'PARTITIONS: {len(self.partitions)}')
        self.decay = numpy.zeros((len(self.partitions), 1))
        for idx, partition in enumerate(decay.keys()):
            self.decay[idx, 0] = decay[partition]
        self.csum = numpy.zeros((len(self.partitions), 1))

        # Setup logging
        self.get_logger().info('Setting up logfiles...')
        self.results = open('results.csv', 'w')
        self.results.write(
            'start,preproc,inference,martingale,end,m_brightness,m_rain,cumsum_brightness,cumsum_rain\n')

        # Prime GPU
        if self.device is not None and self.device != "TFLITE":
            self.get_logger().info('Priming torch device...')
            with torch.no_grad():
                frame = torchvision.transforms.functional.to_tensor(
                    numpy.ones(
                        (self.input_d[0], self.input_d[1], self.n_chan),
                        dtype=numpy.uint8)).half()
                for i in range(20):
                    frame *= 2
                    _, _ = self.encoder(frame.unsqueeze(0).to(self.device))
            if self.device == 'cuda':
                with open('cuda_stats.txt', 'w') as fp:
                    fp.write(torch.cuda.memory_summary())

        # Establish subscribers
        self.get_logger().info('Creating subscribers...')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)
        vid_sub = self.create_subscription(
            CompressedImage,
            '/mock_camera/compressed',
            self.run_detect,
            qos_profile)
        shutdown_sub = self.create_subscription(
            Bool,
            '/mock_camera/done',
            self.stop,
            qos_profile)
        self.get_logger().info('Monolithic detector node setup complete!')

    def run_detect(self, msg: CompressedImage) -> None:
        """Perform OOD detection on an incoming frame.

        Args:
            msg - CompressedImage message from camera.
        """
        start_time = time.time()
        frame = self.preprocess(msg)
        preproc_time = time.time()
        if self.device is not None and self.device != "TFLITE":
            dkl = self.inference(frame)
        else:
            dkl = self.encoder.encode(frame)    # TPU
        inference_time = time.time()
        self.icp(dkl)
        m = self.martingale()
        martingale_time = time.time()
        self.cumsum(m)
        end_time = time.time()
        self.results.write(
            f'{start_time},{preproc_time},{inference_time},{martingale_time},'
            f'{end_time},{numpy.squeeze(m[0])},{numpy.squeeze(m[1])},'
            f'{numpy.squeeze(self.csum[0])},'
            f'{numpy.squeeze(self.csum[1])}\n')

    def preprocess(self, msg: CompressedImage) -> numpy.ndarray:
        """Perform preprocessing on an input frame.  This includes
        decompression, color space conversion and resize operations.

        Args:
            msg - CompressedImage message

        Returns:
            Preprocessed frame as a numpy array.
        """
        frame = numpy.fromstring(bytes(msg.data), numpy.uint8)
        frame = cv2.cvtColor(frame, cv2.IMREAD_COLOR)
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame = torchvision.transforms.functional.resize(
            frame,
            self.input_d,
            self.interpolation)
        if self.n_chan == 1:
            frame = torchvision.transforms.functional.rgb_to_grayscale(frame)
        return frame.half()

    def inference(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Make an inference on the encoder.

        Args:
            frame - raw input data to encoder

        Returns:
            A (1 x n_latent) vector of the frame's KL-divergence for each
            latent dimension.
        """
        with torch.no_grad():
            mu, logvar = self.encoder(frame.unsqueeze(0).to(self.device))
            mu = mu.detach().cpu().numpy()
            logvar = logvar.detach().cpu().numpy()
            return 0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)

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
        p_i = numpy.zeros((len(self.partitions)))
        for partition in range(len(self.partitions)):
            dkl = numpy.sum(dkls * self.z_mask[partition, :])
            p_i[partition] = max(
                numpy.count_nonzero(self.alpha_cal[partition, :] > dkl),
                2) / self.alpha_cal[partition, :].size
        self.past_dkls[:, self.ptr] = numpy.squeeze(p_i)
        self.ptr = (self.ptr + 1) % self.window
        return p_i

    def martingale(self) -> numpy.ndarray:
        """Compute the martingale on the detector's ICP output window.

        Returns:
            A (1 x n_partition) vector of martingale values for all samples
            currently stored in this class's ICP output window.
        """
        m = numpy.zeros((len(self.partitions), 1))
        for partition in range(len(self.partitions)):
            for i in range(100):
                m[partition] += numpy.product(
                    self.eps[i] * numpy.power(
                        self.past_dkls[partition, :], self.eps[i] - 1))
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
        self.get_logger().info('Received message to STOP!')
        self.results.flush()
        self.results.close()
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
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    ood_detector_node = MonolithicDetectorNode(
        args.weights,
        args.alpha_cal,
        args.device,
        args.window)
    try:
        rclpy.spin(ood_detector_node)
    except KeyboardInterrupt:
        print('Monolithic Detector Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Monolithic Detector Received Exception: {str(error)}')
    finally:
        with open('cpu_stats.txt', 'w') as fp:
            fp.write(str(tracemalloc.get_traced_memory()))
        with open('cuda_stats.txt', 'w') as fp:
            fp.write(torch.cuda.memory_summary())
        torch.cuda.empty_cache()
        ood_detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
