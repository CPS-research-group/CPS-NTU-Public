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
        self.last_score = None
        self.sub = rospy.Subscriber(
            'safeduckie6/camera_node/image/compressed',
            CompressedImage,
            self.callback,
            queue_size=1)
        self.sub = rospy.Subscriber(
            'safeduckie6/ood_detector_node/score',
            Float32,
            self.store_last_ood_score,
            queue_size=1)
        rospy.loginfo('Visualizer node setup complete')

    def store_last_ood_score(self, data):
        self.last_score = data.data

    def callback(self, data):
        rospy.loginfo('Recieved a frame')
        frame = cv2.imdecode(
            numpy.frombuffer(data.data, numpy.uint8),
            cv2.IMREAD_COLOR)
        frame = frame[::-1]
        frame = cv2.resize(frame, (640, 480))
        if self.last_score:
            # Draw Rectangle
        cv2.imshow('Visualizer', frame)
        cv2.waitKey(1)
        #rospy.loginfo('Displayed a frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization node.')
    args = parser.parse_args()
    node = Visualizer(node_name='visualizer')
    rospy.spin()

