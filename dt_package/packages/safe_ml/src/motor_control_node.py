#!/usr/bin/env python3
"""Combine control outputs of several nodes to interface with wheels driver
node."""


import os
import yaml
import rospy


from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from std_msgs.msg import Bool, Float32


class MotorControlNode(DTROS):
    """This node takes steering data from lane following and emergency stop
    status from the out-of-distribution detection node.  It uses a PID
    controller to help smooth lane following control inputs and stops the
    Duckiebot immediately if an emergency stop is requested.
    """

    def __init__(self):
        # Call the parent class's contstructor
        super(MotorControlNode, self).__init__(
            node_name="motor_control",
            node_type=NodeType.GENERIC)

        # Load parameters
        params_file = os.path.join(
            os.path.dirname(__file__),
            'motor_control_params.yml')
        params = {}
        with open(params_file, 'r') as paramf:
            params = yaml.safe_load(paramf.read())
        rospy.loginfo(
            'Motor control node launching with the following parameters:')
        for param in params:
            rospy.loginfo(f'{param}: {params[param]}')
        self.params = params

        # Initialize state variables
        self.e_stop = False
        self.angle_integral = 0
        self.last_angle = 0

        # Subscribe to ROS topics
        self.vehicle = os.getenv("VEHICLE_NAME")
        self.wheels_cmd_pub = rospy.Publisher(
            f'/{self.vehicle}/wheels_driver_node/wheels_cmd',
            WheelsCmdStamped,
            queue_size=1)
        self.angle_sub = rospy.Subscriber(
            f'/{self.vehicle}/motor_control_node/angle',
            Float32,
            self.angle_sub_callback,
            queue_size=1)
        self.e_stop_sub = rospy.Subscriber(
            f'/{self.vehicle}/motor_control_node/e_stop',
            Bool,
            self.e_stop_callback,
            queue_size=1)

    def angle_sub_callback(self, angle):
        """Call this method when a new steering angle is delivered to the
        motor_control_node/angle topic.  The angle is fed through a PID
        controller and additional corrections to generate velocities for the
        right and left motors.

        Args:
            angle (Float32): steering angle in radians.
        """
        self.angle_integral += angle.data
        pid = self.params['p'] * angle.data + \
            self.params['i'] * self.angle_integral + \
            self.params['d'] * (angle.data - self.last_angle)
        self.last_angle = angle.data
        msg_wheels_cmd = WheelsCmdStamped()
        msg_wheels_cmd.vel_right = self.params['velocity'] - pid + \
            self.params['right_boost']
        msg_wheels_cmd.vel_left = self.params['velocity'] + pid + \
            self.params['left_boost']
        if not self.e_stop:
            self.wheels_cmd_pub.publish(msg_wheels_cmd)

    def e_stop_callback(self, data):
        """Call this method when a message is received on the
        motor_control_node/e_stop topic.  Right now the data of the message is
        ignored and the both motors will stop immediately on receipt of this
        message.

        Args:
            data (Bool): data from the received e_stop message.
        """
        self.e_stop = True
        msg_wheels_cmd = WheelsCmdStamped()
        msg_wheels_cmd.vel_right = 0.0
        msg_wheels_cmd.vel_left = 0.0
        self.wheels_cmd_pub.publish(msg_wheels_cmd)
        rospy.loginfo('Emergency Stop Activated!')

    def on_shutdown(self):
        """This method is called when ROS needs to shutdown.  The expected
        behavior is that the duckiebot is stopped and all spawned threads are
        given passed a flag to stop execution.
        """
        rospy.loginfo('Lane following node received shutdown signal.')
        self.e_stop = True
        msg_wheels_cmd = WheelsCmdStamped()
        msg_wheels_cmd.vel_right = 0.0
        msg_wheels_cmd.vel_left = 0.0
        self.wheels_cmd_pub.publish(msg_wheels_cmd)


if __name__ == '__main__':
    motor_control_node = MotorControlNode()
    rospy.spin()
