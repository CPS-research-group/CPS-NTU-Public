#!/usr/bin/env python3
"""Light-weight lane following implementation for the DB18."""


import math
import os
import yaml
import cv2
import numpy as np
import rospy


from threading import Event, Lock, Thread
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class LaneFollowingNode(DTROS, Thread):
    """This Node takes in data from the sensors (camera) and outputs commands
    to the wheels.  Due to processing power constraints on the RPi, we are not
    using a series of ROS nodes for each processing step, but managing our own
    thread within this node to try and squeeze better performance on
    low-powered platforms.
    """

    def __init__(self):
        # Call the parent classes' constructors
        super(LaneFollowingNode, self).__init__(
            node_name='lane_following',
            node_type=NodeType.GENERIC)
        Thread.__init__(self)

        # Load parameters
        params_file = os.path.join(
            os.path.dirname(__file__),
            'lane_following_params.yml')
        params = {}
        with open(params_file, 'r') as paramf:
            params = yaml.safe_load(paramf.read())
        rospy.loginfo('Lane following node loaded the following parameters:')
        for param in params:
            rospy.loginfo(f'{param}: {params[param]}')
        self.params = params

        # Subscribe to ROS topics
        self.vehicle = os.getenv("VEHICLE_NAME")
        self.angle_pub = rospy.Publisher(
            f'/{self.vehicle}/motor_control_node/angle',
            Float32,
            queue_size=1)
        self.camera_sub = rospy.Subscriber(
            f'/{self.vehicle}/camera_node/image/compressed',
            CompressedImage,
            self.callback,
            queue_size=1)

        # Create shared buffer and launch the child threads
        self.last_frame = None
        self.lock = Lock()
        self.stop_flag = Event()
        self.start()
        rospy.loginfo('Initialized Image Capture Node')

    def callback(self, image_data):
        """Run this callback function evertime a new compressed image is placed
        on the camera subscriber queue.  Take the compressed image and write it
        to the buffer shared with the lane following thread.

        Args:
            image_data (CompressedImage): compressed image message delivered by
                the camera node.
        """
        if self.lock.acquire():
            try:
                self.last_frame = image_data
            finally:
                self.lock.release()

    def run(self):
        """Run this method in a separate thread.  This thread will read the
        latest image written to the shared buffer and estimate the deviation
        angle from the lane lines.  Angle estimates will be published to the
        motor control node."""
        width = self.params['width']
        height = self.params['height']
        lpf_size = self.params['lpf size']
        lpf = np.zeros(lpf_size)
        lpf_idx = 0
        rate = rospy.Rate(self.params['fps'])
        frame = None
        lane_follower = LaneFollowing(self.params)
        angle = Float32()
        while not rospy.is_shutdown():
            rate.sleep()
            if self.stop_flag.is_set():
                break
            self.lock.acquire()
            try:
                frame = np.fromstring(self.last_frame.data, np.uint8)
            except AttributeError:
                rospy.logwarn('Lane Detector: Camera node is not yet ready...')
                continue
            finally:
                self.lock.release()
            try:
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                frame = cv2.resize(frame, (width, height))
                deviation_angle = lane_follower.detect_lanes(frame)
                lpf[lpf_idx % lpf_size] = deviation_angle
                lpf_idx += 1
                angle.data = np.average(lpf)
                self.angle_pub.publish(angle)
            except LaneDetectionException as err:
                rospy.logerr(f'Error detecting lane lines.... {err}')
        rospy.loginfo('Ending lane following thread loop...')

    def on_shutdown(self):
        """Run this method when ROS initiates a shutdown.  Stop any spawned
        threads."""
        self.stop_flag.set()
        rospy.loginfo('Lane following node is shutting down...')


class LaneDetectionException(Exception):
    """Exception for when an error occurrs during lane detection."""


class LaneFollowing:
    """Class to manage methods related to identifying lanes and estimating
    steering angle.  This class is independent from LaneFollowingNode so that
    lane detection can be run offline outside of a ROS environment for
    debugging purposes.

    Args:
        params (dict): dictionary of parameters used during lane detection.
    """

    def __init__(self, params):
        # This is a pretty ugly constructor, but we're trying to save a few
        # cycles by eliminating all these map lookups for parameters every
        # time lane detection is run.
        self.params = params
        self.width = params['width']
        self.height = params['height']
        self.horizon = int(self.height * params['horizon mask'])
        self.cm_lb = params['color mask']['lower bound']
        self.cm_up = params['color mask']['upper bound']
        self.gb_kernel = (params['gaussian blur'], params['gaussian blur'])
        self.canny_th1 = params['canny']['threshold 1']
        self.canny_th2 = params['canny']['threshold 2']
        self.rho = params['line detection']['rho']
        self.ht_min_thresh = params['line detection']['min threshold']
        self.min_line_length = params['line detection']['min length']
        self.max_line_gap = params['line detection']['max gap']
        self.left_bound = self.width * (1 - params['boundary'])
        self.right_bound = self.width * params['boundary']

    def detect_lanes(self, frame):
        """Detect right and left lane lines present in a camera frame.

        Args:
            frame: numpy array containing raw image data.

        Returns:
            angle (float): deviation angle from center in radians.

        Raises:
            LaneDetectionException if a pair of lane lines could not be found.
        """
        frame = frame[self.horizon:,:]
        frame = cv2.inRange(
            frame,
            np.array(self.cm_lb),
            np.array(self.cm_up))
        frame = cv2.GaussianBlur(frame, self.gb_kernel, 0)
        frame = cv2.Canny(frame, self.canny_th1, self.canny_th2)
        line_segments = cv2.HoughLinesP(
            frame,
            self.rho,
            np.pi / 180,
            self.ht_min_thresh,
            np.array([]),
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap)
        if line_segments is None:
            raise LaneDetectionException("No lanes detected (Hough transform).")
        left_slopes = []
        right_slopes = []
        for segment in line_segments:
            for x1, y1, x2, y2 in segment:
                if y1 == y2:
                    continue
                m = -(x2 - x1) / (y2 - y1)
                if m < 0 and x1 > self.right_bound and x2 > self.right_bound:
                    right_slopes.append(m)
                elif m > 0 and x1 < self.left_bound and x1 < self.left_bound:
                    left_slopes.append(m)
        if len(left_slopes) <= 0 or len(right_slopes) <= 0:
            raise LaneDetectionException("No lanes detected (curve fitting).")
        return math.atan(np.average(left_slopes)) + \
            math.atan(np.average(right_slopes))


if __name__ == '__main__':
    node = LaneFollowingNode()
    rospy.spin()
