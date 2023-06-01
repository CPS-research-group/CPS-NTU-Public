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


sys.path.insert(0, '/ros_ws/src/ecrts/ecrts/yolo')
from models.common import Conv
from models.yolo import Model
from models.experimental import attempt_load, Ensemble
from utils.general import non_max_suppression
from utils.torch_utils import select_device

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

from interfaces.msg import YoloResult
import __main__


class ObjectDetectorNode(Node):

    def __init__(self, weights, size, threshold) -> None:
        super().__init__('object_detector_node')
        self.get_logger().info('Preparing YOLO detector node...')
        self.height = size[0]
        self.width = size[1]
        self.threshold = threshold
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        setattr(__main__, "Ensemble", Ensemble)
        setattr(__main__, "Model", Model)
        #setattr(__main__, "Icp", Icp)
        self.model = self.attempt_load(os.path.join(weights), map_location=self.device)
        self.model.eval()
        self.get_logger().info(f'Have Cuda? {torch.cuda.is_available()}')
        ### Prime GPU ###
        # Still no idea why this needs to be done

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1,
        )
        self.yolo_pub = self.create_publisher(
            YoloResult,
            '/object_detector/yolo',
            qos_profile
        )
        video_sub = self.create_subscription(
            CompressedImage,
            '/mock_camera/compressed',
            self.yolo_detect,
            qos_profile
        )
        shutdown_sub = self.create_subscription(
            Bool,
            '/mock_camera/done',
            self.stop,
            qos_profile
        )
        self.get_logger().info('YOLO node setup complete.')

    def attempt_load(self, weights, map_location):
        """Attempt to load yolov7 model weights.  This is based on the
        attempt_load function in the yolov7 repo, but with the download
        ability removed."""
        model = Ensemble()
        ckpt = torch.load(weights, map_location=map_location)
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())
        for m in model.modules():
            if type(m) in [torch.nn.Hardswish, torch.nn.LeakyReLU, torch.nn.ReLU, torch.nn.ReLU6, torch.nn.SiLU]:
                m.inplace = True
            elif type(m) is torch.nn.Upsample:
                m.recompute_scale_factor = None
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()

        if len(model) == 1:
            return model[-1]
        else:
            rospy.loginfo('Ensemble created with %s\n' % weights)
            for k in ['names', 'stride']:
                setattr(model, k, getattr(model[-1], k))
            return model

    def yolo_detect(self, msg: CompressedImage) -> None:
        start = self.get_clock().now().to_msg()
        yolo_result = YoloResult()
        yolo_result.start = start
        x = numpy.fromstring(bytes(msg.data), numpy.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = torchvision.transforms.functional.to_tensor(x)
        x = torchvision.transforms.functional.resize(x, (self.height, self.width))
        x.type(torch.float32)
        x = x[None, :]
        results = self.model.forward(x, augment=True)
        results = non_max_suppression(results[0], conf_thres=self.threshold, iou_thres=0.5)
        yolo_result.end = self.get_clock().now().to_msg()
        self.yolo_pub.publish(yolo_result)

    def stop(self, msg: Bool) -> None:
        self.get_logger().warn('Recieved message to STOP!')
        raise Exception('Received STOP message.')

def main():
    parser = argparse.ArgumentParser('OOD Detector Node')
    parser.add_argument(
        '-w',
        '--weights',
        help='weights file for the YOLO detector'
    )
    parser.add_argument(
        '--height',
        help='YOLO input height'
    )
    parser.add_argument(
        '--width',
        help='YOLO input width'
    )
    parser.add_argument(
        '--threshold',
        help='Detection threshold'
    )
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    object_detector_node = ObjectDetectorNode(
        weights=args.weights,
        size=(int(args.height), int(args.width)),
        threshold=float(args.threshold),
    )
    try:
        rclpy.spin(object_detector_node)
    except KeyboardInterrupt:
        print('YOLO Node Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'YOLO Node generated exception: {str(error)}')
    finally:
        object_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
