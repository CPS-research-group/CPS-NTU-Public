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
from glob import glob


FPS = 5.0
DIM = (149, 117)
VIDEO = '~/ood_rain.mp4'


class MockCameraImgNode(Node):
    """Node that simulates messages coming from Duckietown's camera by playing
    a stream of a video recording."""

    def __init__(self, fps, dimensions, video_file=None, img_folder=None):
        super().__init__('mock_camera_node')
        self.get_logger().info('Preparing mock camera node...')
        self.dimensions = dimensions
        self.video_f = video_file
        self.img_files = []
        if video_file is not None:
            self.vid_in = cv2.VideoCapture(os.path.expanduser(video_file))
            if not self.vid_in.isOpened():
                self.get_logger().error('Could not open file, is path correct?')
                return
        else:
            for imagefile in sorted(glob(img_folder + "/*.png")):
                # Check if image filename is standard (name.png)
                if len(imagefile.split("/")[-1].split(".")) == 2:
                    self.img_files.append(imagefile)

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
        if self.video_f is not None:
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
                return
            img = cv2.resize(img, self.dimensions)
            retval, compressed = cv2.imencode('.jpg', img)
            if not retval:
                self.get_logger().error('Failed to compress image...')
                return
        else:
            data_frame.header.frame_id = 'Mock Camera Image'
            data_frame.format = 'jpeg'
            try:
                current_image_file = self.img_files.pop(0)
                # print(current_image_file)
                current_image = cv2.imread(current_image_file)
                retval, compressed = cv2.imencode('.jpg', current_image)

                if not retval:
                    self.get_logger().error('Failed to encode image...')
                    return
            except IndexError as e:
                self.get_logger().warn('No more images left in sequence...')
                self.timer.destroy()
                time.sleep(10)
                self.get_logger().warn('Sending STOP command...')
                msg = Bool()
                msg.data = True
                self.stop_pub.publish(msg)
                self.executor.shutdown(10)
                raise Exception("Camera Shutdown!!")
                # return
                # else:
                #     raise Exception("End of image sequence!")

        data_frame.header.stamp = self.get_clock().now().to_msg()
        data_frame.data = bytes(compressed.flatten().tolist())
        self.vid_pub.publish(data_frame)


def main():
    parser = argparse.ArgumentParser('Mock Camera Image Node')
    parser.add_argument(
        '-v',
        '--video',
        help='video file to read from')
    parser.add_argument(
        '-f',
        '--folder',
        help='folder to read images from')
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
        mock_camera_img_node = MockCameraImgNode(args.fps, DIM, video_f, None)
    if args.folder:
        print(f'loading: {args.folder}')
        img_f = args.folder
        rclpy.init(args=sys.argv)
        mock_camera_img_node = MockCameraImgNode(args.fps, DIM, None, img_f)
    try:
        rclpy.spin(mock_camera_img_node)
    except KeyboardInterrupt:
        print('Mock Camera Received Keyboard Interrupt, shutting down...')
    except Exception as error:
        print(f'Received Exception: {str(error)}')
    finally:
        mock_camera_img_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
