"""Launch all nodes with 1 executor."""

import argparse
import sys
import rclpy
import traceback

from .preproc_encoder_node import PreprocEncoderNode
#from .mgale_cumsum_node import MartingaleCumSumNode
from .mgale_all_node import MartingaleCumSumNode
from .mock_camera_node import MockCameraNode
from .mock_camera_img_node import MockCameraImgNode
from .monolithic_detector_node_f16 import MonolithicDetectorNode
from .bi3dof_ood_detector_node_f16 import OFDetectorNode

from rclpy.executors import Executor, ExternalShutdownException, TimeoutException
from rclpy import logging

from concurrent.futures import ThreadPoolExecutor
import time


class DebugExecutor(Executor):

    def __init__(self, num_threads: int = None, *, context=None) -> None:
        super().__init__(context=context)
        self._executor = ThreadPoolExecutor(num_threads)

    def spin_once(self, timeout_sec: float = None) -> None:
        try:
            handler, entity, node = self.wait_for_ready_callbacks(
                timeout_sec=timeout_sec)
        except ExternalShutdownException:
            pass
        except ShutdownException:
            pass
        except TimeoutException:
            pass
        else:
            logging.get_logger('executor').info(
                f'Ready Set is {[x[2] for x in self._tasks]} @ {time.time()}')
            logging.get_logger('executor').info(
                f'There are {self._work_tracker._num_work_executing} tasks in progress.')
            logging.get_logger('executor').info(
                f'Submitted {node} for execution @ {time.time()}')
            self._executor.submit(handler)


def main():
    parser = argparse.ArgumentParser('Launch everything.')
    parser.add_argument(
        '--weights',
        help='weights file.')
    parser.add_argument(
        '--alpha_cal',
        required=False,
        help='alpha cal file.')
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda', 'tpu', 'tflite'],
        default='cuda',
        help='torch device.')
    parser.add_argument(
        '--window',
        type=int,
        default=20,
        help='Martingale window size.')
    parser.add_argument(
        '--decay',
        type=float,
        default=20,
        help='Decay term in cumsum.')
    parser.add_argument(
        '--video',
        help='video file input.')
    parser.add_argument(
        '--executor',
        choices=['st', 'mt'],
        default='st',
        help='single threaded or multi-threaded executor')
    parser.add_argument(
        '--arch',
        choices=['monolithic', 'chain', 'mono_of'],
        help='Task graph for OOD detector')
    parser.add_argument(
        '--fps',
        type=float,
        help='FPS for video source')
    parser.add_argument(
        '--get_mem',
        action='store_true',
        default=False,
        help='Get Python memory stats')
    args = parser.parse_args()
    rclpy.init(args=sys.argv)
    if args.arch == 'chain':
        preproc_encoder_node = PreprocEncoderNode(
            args.weights,
            args.alpha_cal,
            args.device)
        # mgale_rain_node = MartingaleCumSumNode(
        #    args.weights,
        #    args.alpha_cal,
        #    args.device,
        #    args.window,
        #    'rain',
        #    'rain_martingale')
        # mgale_brightness_node = MartingaleCumSumNode(
        #    args.weights,
        #    args.alpha_cal,
        #    args.device,
        #    args.window,
        #    'brightness',
        #    'brightness_martingale')
        mgale_node = MartingaleCumSumNode(
            args.weights,
            args.alpha_cal,
            args.device,
            args.window,
            'martingale')
        #mock_camera_node = MockCameraNode(args.fps, (224, 224), args.video)
        if args.executor == 'mt':
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
            #executor = DebugExecutor(num_threads=2)
            print('Launching with MultiThreadedExecutor')
        else:
            executor = rclpy.executors.SingleThreadedExecutor()
            print('Launching with SingleThreadedExecutor')
        executor.add_node(preproc_encoder_node)
        # executor.add_node(mgale_rain_node)
        # executor.add_node(mgale_brightness_node)
        executor.add_node(mgale_node)
        # executor.add_node(mock_camera_node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            print('RECEIVED KEYBOARD INTERUPT')
        except Exception as error:
            print('Executor Received Execption...')
            print(str(error))
        finally:
            preproc_encoder_node.destroy_node()
            # mgale_rain_node.destroy_node()
            # mgale_brightness_node.destroy_node()
            # mock_camera_node.destroy_node()
            mgale_node.destroy_node()
            rclpy.shutdown()
    elif args.arch == 'monolithic':
        monolithic_ood_node = MonolithicDetectorNode(
            args.weights,
            args.alpha_cal,
            args.device,
            args.window)
        #mock_camera_node = MockCameraNode(args.fps, (224, 224), args.video)
        if args.executor == 'mt':
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
            #executor = rclpy.DebugExecutor(num_threads=4)
            print('Launching with MultiThreadedExecutor')
        else:
            executor = rclpy.executors.SingleThreadedExecutor()
            print('Launching with SingleThreadedExecutor')
        executor.add_node(monolithic_ood_node)
        # executor.add_node(mock_camera_node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            print('RECEIVED KEYBOARD INTERRUPT')
        except Exception as error:
            print('Executor Received Exception...')
            print(str(error))
        finally:
            monolithic_ood_node.destroy_node()
            # mock_camera_node.destroy_node()
            rclpy.shutdown()
        print('Outside exception handler...')
    elif args.arch == 'mono_of':
        print('Launching OF Processes')
        mono_of_ood_node = OFDetectorNode(args)
        # mock_camera_img_node = MockCameraImgNode(args.fps, (160, 120), img_folder=args.video)
        if args.executor == 'mt':
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
            print('Launching with MultiThreadedExecutor')
        else:
            executor = rclpy.executors.SingleThreadedExecutor()
            print('Launching with SingleThreadedExecutor')
        executor.add_node(mono_of_ood_node)
        # executor.add_node(mock_camera_img_node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            print('RECEIVED KEYBOARD INTERRUPT')
        except Exception as error:
            print('Executor Received Exception...')
            print(str(error))
        finally:
            mono_of_ood_node.destroy_node()
            # mock_camera_img_node.destroy_node()
            rclpy.shutdown()
        print('Outside exception handler...')


if __name__ == '__main__':
    main()
