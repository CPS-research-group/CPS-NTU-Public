#!/usr/bin/env python3


import argparse
import os
import sys
import time


import cv2
import numpy
import rospy
import torch
import torchvision


from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov7'))
from models.common import Conv
from models.yolo import Model 
from models.experimental import attempt_load, Ensemble
from utils.general import non_max_suppression
from utils.torch_utils import select_device


class ObjectDetector(DTROS):
    """Object detection node.

    Args:
        node_name - unique ROS1 node name
        size - integer size in pixels of the input image
    """

    def __init__(self, node_name, size):
        super(ObjectDetector, self).__init__(
            node_name=node_name,
            node_type=NodeType.GENERIC)
        self.size = size
        self.weights = torch.load(
            os.path.join(os.path.dirname(__file__),
            f'best{self.size}.pt'))
        device = select_device('cpu')
        self.model = self.attempt_load(
            os.path.join(os.path.dirname(__file__), f'best{self.size}.pt'),
            map_location=device)
        self.model.float()
        torch.backends.quantized.engine = 'qnnpack'
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8)
        self.sub = rospy.Subscriber(
            'safeduckie6/camera_node/image/compressed',
            CompressedImage,
            self.callback)
        rospy.loginfo('Object detection node setup complete')

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


    def callback(self, data):
        """Callback function when an image is published by the camera.
        
        Args:
            data - ROS CompressedImage data
        """
        rospy.loginfo('YOLO: Recieved an image...')
        frame = cv2.imdecode(
            numpy.frombuffer(data.data, numpy.uint8),
            cv2.IMREAD_COLOR)
        frame = frame[::-1]
        frame = cv2.resize(frame, (self.size, self.size))
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame.type(torch.float32)
        frame = frame[None, :]
        results = self.model.forward(frame, augment=True)
        results = non_max_suppression(results[0], conf_thres=0.1, iou_thres=0.5)
        rospy.loginfo('YOLO: Done...')
        rospy.loginfo(f'{results}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Object detection node.')
    parser.add_argument(
            '--size',
            help='YOLO input size',
            type=int,
            choices=[64, 96, 128, 160])
    args = parser.parse_args()
    rospy.loginfo('Starting object detector')
    node = ObjectDetector(node_name='object_detector', size=args.size)
    rospy.spin()

