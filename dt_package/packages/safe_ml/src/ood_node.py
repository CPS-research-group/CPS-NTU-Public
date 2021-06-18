#!/usr/bin/env python3
"""Module containing OOD detection node."""


import os
from threading import Event, Lock, Thread


import cv2
import numpy as np
import yaml
import rospy
import torch
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import Bool
from sensor_msgs.msg import CompressedImage
from ood_module_betav import Detector


class OodNode(DTROS, Thread):
    """This Node subscribes to the camera feed and determines if an
    out-of-distribution condition has occurred.  If it has, it performs
    some control action to safe the duckiebot.
    """

    def __init__(self):
        # Initialize parent classes' constructors
        super(OodNode, self).__init__(
            node_name='ood_node',
            node_type=NodeType.GENERIC)
        Thread.__init__(self)

        # Load parameters
        params_file = os.path.join(
            os.path.dirname(__file__),
            'ood_params.yml')
        params = {}
        with open(params_file, 'r') as paramf:
            params = yaml.safe_load(paramf.read())
        self.params = params
        self.modelfile = params["model"]
        self.modelfile = os.path.dirname(__file__) + "/" + self.modelfile
        self.torchdevice = params["torch_device"]
        self.model = Detector(self.modelfile, torch.device(self.torchdevice))

        # Initialize shared memory
        self.last_frame = None
        self.lock = Lock()
        self.stop_flag = Event()

        # Subscribe to ROS topics
        self.vehicle = os.getenv("VEHICLE_NAME")
        self.e_stop_pub = rospy.Publisher(
            f'/{self.vehicle}/motor_control_node/e_stop',
            Bool,
            queue_size=1)
        self.sub = rospy.Subscriber(
            f'/{self.vehicle}/camera_node/image/compressed',
            CompressedImage,
            self.callback)

        # Start child threads
        self.start()
        rospy.loginfo(
            f'OOD Module successfully initialized with {self.modelfile}, using'
            f'{self.torchdevice}')

    def callback(self, data):
        """This method is called whenever a new image is published by the
        camera node.  It forwards the image to a shared buffer with another
        thread.

        Args:
            data (CompressedImage) - compressed image data provided by the
                duckietown camera node.
        """
        if self.lock.acquire(True, timeout=0.02):
            try:
                self.last_frame = data
            finally:
                self.lock.release()

    def run(self):
        """This method runs in a separate thread and takes care of the OOD
        detection.  It accesses images published to a shared buffer to ensure
        that it always has the latest image from the camera before processing.
        """
        frame = None
        stamp = None
        while not rospy.is_shutdown():
            if self.stop_flag.is_set():
                break
            self.lock.acquire()
            try:
                frame = np.fromstring(self.last_frame.data, np.uint8)
                stamp = self.last_frame.header.stamp
            except AttributeError:
                rospy.logwarn('OOD: Camera node is not yet ready...')
                continue
            finally:
                self.lock.release()
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
            result = self.model.check(frame)
            rospy.loginfo(f'OOD Finished: value={result["value"]} im_time={stamp.to_sec()}')
            if result['isood']:
                e_stop_msg = Bool()
                e_stop_msg.data = True
                #rospy.loginfo(f'OOD Finished: value={result["value"]} im_time={stamp.to_sec()}')
                self.e_stop_pub.publish(e_stop_msg)
        rospy.loginfo('Ending OOD thread...')

    def on_shutdown(self):
        """Run this method when ROS initiates a shutdown.  Stop any spawned
        threads."""
        self.stop_flag.set()
        rospy.loginfo('OOD node is shutting down...')


if __name__ == '__main__':
    node = OodNode()
    rospy.spin()
