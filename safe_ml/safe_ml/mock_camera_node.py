import argparse
import os
import sys
import time
import threading


import cv2
import numpy
import rclpy


from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy, QoSPresetProfiles
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import CompressedImage


class MockCameraNode(Node):
    """Node that simulates messages coming from Duckietown's camera by playing
    a stream of a video recording."""

    def __init__(self):
        super().__init__('mock_camera_node')
        self.get_logger().info('Preparing mock camera node...')
        self.declare_parameter('height', 600)
        self.declare_parameter('width', 800)
        self.declare_parameter('fps', 20)
        self.frame_counter = 0
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_ALL,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE)
        self.vid_pub = self.create_publisher(
            CompressedImage,
            '/mock_camera/compressed',
            qos_profile)
        self.stop_pub = self.create_publisher(
            Bool,
            '/mock_camera/done',
            qos_profile)
        fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.timer = self.create_timer(1 / fps, self.publish_frame)
        self.get_logger().info('Mock camera setup complete...')

    def publish_frame(self):
        data_frame = CompressedImage()
        data_frame.header = Header()
        data_frame.header.frame_id = 'RGB Video Stream'
        data_frame.format = 'jpeg'
        self.frame_counter += 1
        if self.frame_counter >= 1000:
            self.timer.destroy()
            self.get_logger().warn('Sending STOP command...')
            msg = Bool()
            msg.data = True
            self.stop_pub.publish(msg)
            self.executor.shutdown(10)
            raise Exception("Camera Shutdown!!")
        height = self.get_parameter('height').get_parameter_value().integer_value
        width = self.get_parameter('width').get_parameter_value().integer_value
        img = (255 * numpy.random.rand(height, width, 3)).astype(numpy.uint8)
        retval, compressed = cv2.imencode('.jpg', img)
        if not retval:
            self.get_logger().error('Failed to compress image...')
            raise Exception("Camera Shutdown!!")
        data_frame.header.stamp = self.get_clock().now().to_msg()
        data_frame.data = bytes(compressed.flatten().tolist())
        self.vid_pub.publish(data_frame)


def main():
    rclpy.init(args=sys.argv)
    mock_camera_node = MockCameraNode()
    try:
        rclpy.spin(mock_camera_node)
    except KeyboardInterrupt:
        print('Mock Camera Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Mock Camera Received Exception: {str(error)}')
    finally:
        mock_camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()



