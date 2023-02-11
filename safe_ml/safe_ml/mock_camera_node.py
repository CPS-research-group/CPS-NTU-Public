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


FPS = 5.0
DIM = (640, 480)
VIDEO = '~/ood_rain.mp4'


class MockCameraNode(Node):
    """Node that simulates messages coming from Duckietown's camera by playing
    a stream of a video recording."""

    def __init__(self, fps, dimensions, video_file):
        super().__init__('mock_camera_node')
        self.get_logger().info('Preparing mock camera node...')
        self.dimensions = dimensions
        self.vid_in = cv2.VideoCapture(os.path.expanduser(video_file))
        if not self.vid_in.isOpened():
            self.get_logger().error('Could not open file, is path correct?')
            return
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
        self.end_time = None
        self.timer = self.create_timer(1 / fps, self.publish_frame)
        self.get_logger().info('Mock camera setup complete...')

    def publish_frame(self):
        data_frame = CompressedImage()
        data_frame.header = Header()
        data_frame.header.frame_id = 'Mock Camera Image'
        data_frame.format = 'jpeg'
        retval, img = self.vid_in.read()
        if not retval:
            self.timer.destroy()
            self.get_logger().warn('No more images left in sequence..')
            time.sleep(10)
            self.get_logger().warn('Sending STOP command...')
            msg = Bool()
            msg.data = True
            self.stop_pub.publish(msg)
            self.executor.shutdown(10)
            raise Exception("Camera Shutdown!!")
            # return
        img = cv2.resize(img, self.dimensions)
        retval, compressed = cv2.imencode('.jpg', img)
        if not retval:
            self.get_logger().error('Failed to compress image...')
            raise Exception("Camera Shutdown!!")
            # return
        data_frame.header.stamp = self.get_clock().now().to_msg()
        data_frame.data = bytes(compressed.flatten().tolist())
        self.vid_pub.publish(data_frame)


def main():
    parser = argparse.ArgumentParser('Mock Camera Node')
    parser.add_argument(
        '-v',
        '--video',
        help='video file to read from')
    parser.add_argument(
        '--fps',
        type=float,
        help='FPS for video source')
    args = parser.parse_args()
    video_f = VIDEO
    if args.video:
        print(f'loading file: {args.video}')
        video_f = args.video
    rclpy.init(args=sys.argv)
    mock_camera_node = MockCameraNode(args.fps, DIM, video_f)
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
