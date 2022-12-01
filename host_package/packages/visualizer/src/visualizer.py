#!/usr/bin/env python3


import argparse
import os
import sys
import time


import cv2
import numpy
import rospy


from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class Visualizer(DTROS):
    """Visualize what the Duckiebot sees.

    Args:
        node_name - ROS1 node name
    """

    def __init__(self, node_name):
        super(Visualizer, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.sub = rospy.Subscriber(
            'safeduckie6/camera_node/image/compressed',
            CompressedImage,
            self.callback,
            queue_size=1)
        rospy.loginfo('Visualizer node setup complete')

    def callback(self, data):
        frame = cv2.imdecode(
            numpy.frombuffer(data.data, numpy.uint8),
            cv2.IMREAD_COLOR)
        frame = frame[::-1]
        cv2.imshow('Visualizer', frame)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization node.')
    args = parser.parse_args()
    node = Visualizer(node_name='visualizer')
    rospy.spin()

