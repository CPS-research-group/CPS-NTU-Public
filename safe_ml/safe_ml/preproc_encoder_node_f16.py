"""ROS2 Node for for the preproprocessing and encoder inference in the
BetaVAE OOD Detector."""

import argparse
import io
import json
import os
import pkgutil
import sys
import time

import cv2
import numpy
import rclpy
import torch
import torchvision
import tflite_runtime.interpreter as tflite


from .bvae import Encoder, Encoder_TPU, Encoder_TFLITE_BVAE
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSPresetProfiles)
from std_msgs.msg import Bool, Float64MultiArray, Header, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import CompressedImage
from ament_index_python.packages import get_package_share_directory

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
except:
    pass


class PreprocEncoderNode(Node):
    """Preprocessing and Encoder.

    Args:
        weights - weights file to use for the encoder.
        alpha_cal - alpha calibration set file to use for ICP.
        device - currently on 'CUDA' or 'CPU' supported.
    """

    def __init__(self,
                 weights: str,
                 alpha_cal: str,
                 device: str) -> None:
        # ROS2 node initialization
        super().__init__('preproc_encoder_node')
        self.get_logger().info('Preparing preproc/encoder detector node...')

        # Load calibration data
        self.get_logger().info('Loading calibration data...')
        print('ALPHA CAL:')
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

        # Setup logging
        self.get_logger().info('Setting up logfiles...')
        self.results = open('results_preproc_encoder.csv', 'w')
        self.results.write('start,preproc,inference\n')

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
                    self.get_logger().info('Prime N')
                    _, _ = self.encoder(frame.unsqueeze(0).to(self.device))
            if self.device == 'cuda':
                with open('cuda_stats.txt', 'w') as fp:
                    fp.write(torch.cuda.memory_summary())

        # Establish publishers
        self.get_logger().info('Creating publishers...')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)
        self.dkl_pub = self.create_publisher(
            Float64MultiArray,
            '/ood_detector/dkls',
            qos_profile)

        # Establish subscribers
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
        self.send_msg = Float64MultiArray()
        self.send_msg.layout = MultiArrayLayout()
        self.send_msg.layout.data_offset = 0
        dim = MultiArrayDimension()
        dim.label = 'x'
        dim.size = self.n_latent
        dim.stride = 1
        self.send_msg.layout.dim = [dim]

        self.get_logger().info('Preprocessing and encoder node setup complete!')

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
            dkl = self.encoder.encode(frame)    # TPU / TFLITE
        inference_time = time.time()
        self.send_msg.data = dkl.flatten().tolist()
        self.dkl_pub.publish(self.send_msg)
        self.results.write(f'{start_time},{preproc_time},{inference_time}\n')

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
        frame = torchvision.transforms.functional.resize(frame, self.input_d, self.interpolation)
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

    def stop(self, msg: Bool) -> None:
        """Stop this node."""
        self.get_logger().warn('Received message to STOP!')
        self.results.flush()
        self.results.close()
        raise Exception('Received STOP message.')


def main():
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
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    ood_detector_node = PreprocEncoderNode(
        args.weights,
        args.alpha_cal,
        args.device)
    try:
        rclpy.spin(ood_detector_node)
    except KeyboardInterrupt:
        print('Preproc Encoder Node Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Preproc Encoder Node Received Exception: {str(error)}')
    finally:
        if args.device == 'cuda':
            with open('cuda_stats.txt', 'w') as fp:
                fp.write(torch.cuda.memory_summary())
            torch.cuda.empty_cache()
        ood_detector_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
