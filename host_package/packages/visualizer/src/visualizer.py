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
            self.camera_callback,
            queue_size=1)
        self.sub = rospy.Subscriber(
            'safeduckie6/ood_detector_node/score',
            Float32,
            self.ood_score_callback,
            queue_size=1)
        rospy.loginfo('Visualizer node setup complete')

    def ood_score_callback(self, data):
        """Display the last frame and its associated OOD score as a bar in the
        upper left hand corner of the image."""
        if not self.last_frame:
            return
        rospy.loginfo('Recieved a score')
        frame = cv2.imdecode(
            numpy.frombuffer(self.last_frame, numpy.uint8),
            cv2.IMREAD_COLOR)
        frame = frame[::-1]
        frame = cv2.resize(frame, (640, 480))
        cv2.rectangle(
            frame,
            (10, 10),
            (30, int(data.data) + 10),
            (0, 255, 0),
            -1)
        cv2.imshow('Visualizer', frame)
        cv2.waitkey(1)

    def camera_callback(self, data):
        """Store a camera image for display when its OOD score arrives."""
        rospy.loginfo('Recieved a frame')
        self.last_frame = data


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualization node.')
    args = parser.parse_args()
    node = Visualizer(node_name='visualizer')
    rospy.spin()

