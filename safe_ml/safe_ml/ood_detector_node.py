"""Monolithic OOD Detection Node."""

import argparse
import json
import os
import pkgutil
import sys


import cv2
import numpy
import rclpy
import torch
import torchvision


sys.path.insert(0, '/ros_ws/src/ecrts/ecrts/ood')
from ood_detector import OodDetector
from vae import Vae
from icp import Icp

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

from interfaces.msg import OodScore
import __main__


class OodDetectorNode(Node):

    def __init__(self, weights, size) -> None:
        super().__init__('ood_detector_node')
        self.get_logger().info('Preparing OOD detector node...')
        self.height = size[0]
        self.width = size[1]
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        setattr(__main__, "OodDetector", OodDetector)
        setattr(__main__, "Vae", Vae)
        setattr(__main__, "Icp", Icp)
        self.ood_detector = torch.load(weights, map_location=self.device)
        self.ood_detector.eval()
        self.get_logger().info(f'Have Cuda? {torch.cuda.is_available()}')
        ### Prime GPU ###
        # Still no idea why this needs to be done

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1,
        )
        self.oodscore_pub = self.create_publisher(
            OodScore,
            '/ood_detector/oodscore',
            qos_profile
        )
        video_sub = self.create_subscription(
            CompressedImage,
            '/mock_camera/compressed',
            self.check_ood,
            qos_profile
        )
        shutdown_sub = self.create_subscription(
            Bool,
            '/mock_camera/done',
            self.stop,
            qos_profile
        )
        self.get_logger().info('OOD detector node setup complete.')

    def check_ood(self, msg: CompressedImage) -> None:
        start = self.get_clock().now().to_msg()
        oodscore = OodScore()
        oodscore.start = start
        x = numpy.fromstring(bytes(msg.data), numpy.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = torchvision.transforms.functional.to_tensor(x)
        x = torchvision.transforms.functional.resize(x, (self.height, self.width))
        oodscore.oodscore = float(self.ood_detector.forward(x.unsqueeze(0)).detach())
        oodscore.end = self.get_clock().now().to_msg()
        self.oodscore_pub.publish(oodscore)

    def stop(self, msg: Bool) -> None:
        self.get_logger().warn('Recieved message to STOP!')
        raise Exception('Received STOP message.')

def main():
    parser = argparse.ArgumentParser('OOD Detector Node')
    parser.add_argument(
        '-w',
        '--weights',
        help='weights file for the OOD detector'
    )
    parser.add_argument(
        '--height',
        help='OOD detector input height'
    )
    parser.add_argument(
        '--width',
        help='OOD detector input width'
    )
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    ood_detector_node = OodDetectorNode(
        weights=args.weights,
        size=(int(args.height), int(args.width))
    )
    try:
        rclpy.spin(ood_detector_node)
    except KeyboardInterrupt:
        print('OOD Detector Node Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'OOD Detector Node generated exception: {str(error)}')
    finally:
        ood_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
