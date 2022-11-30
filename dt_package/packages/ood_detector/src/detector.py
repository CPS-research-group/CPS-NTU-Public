#!/usr/bin/env python3


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


sys.path.insert(0, os.path.dirname(__file__))
from vae import VaeEncoder


class OodDetector(DTROS):
    """OOD detection node.

    Args:
        node_name - ROS1 node name
    """

    def __init__(self, node_name):
        super(OodDetector, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.model = torch.load(os.path.join(os.path.dirname(__file__), 'vae_10flows_78latents_9862beta.pt_enc.pt'))
        #self.model = VaeEncoder(
        #    (60, 80),
        #    10,
        #    30,
        #    10000,
        #    1,
        #    0)
        torch.backends.quantized.engine = 'qnnpack'
        #self.model.float()
        self.model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8)
        self.model.eval()
        self.buf = [None] * self.model.n_frames
        self.ptr = 0
        self.last_frame = None
        self.flow = None
        self.count = 0
        self.sub = rospy.Subscriber(
            'safeduckie6/camera_node/image/compressed',
            CompressedImage,
            self.callback)
        rospy.loginfo('OOD detector node setup complete')

    def callback(self, data):
        rospy.loginfo('OOD Detector: Recieved an image...')
        ret = numpy.zeros((
            self.model.input_d[0],
            self.model.input_d[1],
            self.model.n_frames))
        frame = cv2.imdecode(
            numpy.frombuffer(data.data, numpy.uint8),
            cv2.IMREAD_COLOR)
        frame = frame[::-1]
        frame = cv2.resize(
            frame,
            (self.model.input_d[1], self.model.input_d[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.last_frame is not None:
            self.flow = cv2.calcOpticalFlowFarneback(
                self.last_frame,
                frame,
                self.flow,
                pyr_scale=0.5,
                levels=1,
                iterations=1,
                winsize=15,
                poly_n=5,
                poly_sigma=1.1,
                flags=0 if self.flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW)
            self.buf[self.ptr] = numpy.copy(self.flow[:, :, 0])
            if self.count >= self.model.n_frames - 1:
                for i in range(self.model.n_frames):
                    ret[:, :, i] = self.buf[self.ptr]
                    self.ptr = (self.ptr + 1) % self.model.n_frames
            self.ptr = (self.ptr + 1) % self.model.n_frames
            self.count += 1
        self.last_frame = numpy.copy(frame)
        sample = torch.from_numpy(ret)
        sample = torch.swapaxes(sample, 1, 2)
        sample = torch.swapaxes(sample, 0, 1)
        sample = sample.nan_to_num(0)
        sample = ((sample + 64) / 128).clamp(0, 1)
        sample = sample.type(torch.FloatTensor)
        with torch.no_grad():
            mu, logvar = self.model(sample.unsqueeze(0))
            ood_score = float((0.5 * mu.pow(2) + logvar.exp() - logvar - 1).sum())
            rospy.loginfo('OOD Detector: Got OOD score: ' + str(ood_score))


if __name__ == '__main__':
    node = OodDetector(node_name='ood_detector')
    rospy.spin()

