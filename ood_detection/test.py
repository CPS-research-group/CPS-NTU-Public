#!/usr/bin/env python3
"""Run BetaVAE OOD detector on a text video and generate detection results
and timings."""

from quantization.dynamic_quantisation import dynamic_quantise
from knowledge_distilation.student_arch_1 import ModEncoder
from quantization.static_quantisation import static_quantise
from knowledge_distilation.student_arch_2 import Mod2BetaVae, Mod2Encoder
from quantization.quantization_aware_training import qat_quantise

from typing import Tuple
import argparse
import json
import os
import re
import time
import warnings
from xmlrpc.client import Boolean
import cv2
import numpy
import numpy.matlib
import pandas
import torch
import torchvision
from quantization.quant_bvae import Encoder


warnings.filterwarnings('ignore')


class BetaVAEDetector:
    """This class represents and end-to-end BetaVAE OOD Detector.
    
    Args:
        weights - path to weights file to use in the encoder.
        alpha_cal - path to alpha calibration set (must be in json format,
            see calibration.py)
        window - length of Martingale window in samples.
        decay - the decay constant to use in the cummulative summation
            algorithm.
    """

    def __init__(self,
                 weights: str,
                 alpha_cal: str,
                 window: int,
                 decay: float, quantize = None) -> None:
        # Load calibration data
        self.partition_map = []
        with open(alpha_cal, 'r') as alpha_cal_f:
            cal_data = json.loads(alpha_cal_f.read())
            self.n_latent = cal_data['PARAMS']['n_latent']
            self.n_chan = cal_data['PARAMS']['n_chan']
            self.input_d = tuple(cal_data['PARAMS']['input_d'])
            del cal_data['PARAMS']
            self.alpha_cal = numpy.zeros((
                len(cal_data.keys()),
                len(list(cal_data.values())[0]['dkls'])))
            self.z_mask = numpy.zeros((len(cal_data.keys()), self.n_latent))
            for idx, item in enumerate(cal_data.items()):
                self.partition_map.append(item[0])
                self.z_mask[idx, item[1]['top_z'][0][0]] = 1
                self.alpha_cal[idx, :] = numpy.sum(
                    numpy.array(item[1]['dkls']) * self.z_mask[idx, :],
                    axis=1)

        # Setup encoder network
        self.device = torch.device('cpu')
        self.encoder = ModEncoder(
            n_latent=self.n_latent,
            n_chan=self.n_chan,
            input_d=self.input_d)
        
        if quantize == 'dq':
            self.encoder = dynamic_quantise(self.encoder)

        elif quantize == 'sq':
            self.encoder = static_quantise(self.encoder)

        elif quantize == 'qat':
            self.encoder = qat_quantise(self.encoder)

        self.encoder.load_state_dict(torch.load(weights, map_location=torch.device('cpu') ))
        self.encoder.eval()
        self.encoder.to(self.device)
        
        # Setup martingale
        self.ptr = 0
        self.window = window
        self.past_dkls = numpy.zeros((len(self.partition_map), self.window))
        self.eps = numpy.linspace(0, 1, 100)

        # Setup cumsum
        self.decay = decay
        self.csum = numpy.zeros((len(self.partition_map), 1))

        # Setup logging frames
        self.timing_df = pandas.DataFrame(
            columns=['start', 'preproc', 'inference', 'martingale', 'end'])
        self.detect_dfs = {}
        for partition in self.partition_map:
            self.detect_dfs[partition] = pandas.DataFrame(
                columns=['p_i', 'm', 'cumsum', 'gt'])

    def run_detect(self, video: str) -> None:
        """Perform OOD detection on all frames in a video sequentially.
        
        Args:
            video - path to video file.  The video file must be named
                according to the format in prep_vids.py to ensure that the
                ground truth labels are generated correctly.
        """
        reader = cv2.VideoCapture(video)
        vid_length = reader.get(cv2.CAP_PROP_FRAME_COUNT)
        ood_start = -1
        ood_end = -1
        match = re.search('.*_m(\d).*', video)
        if match:
            match_val = int(match.groups()[0])
            if match_val == 1:
                ood_start = 0
                ood_end = vid_length
            elif match_val == 2:
                ood_start = int(0.5 * vid_length)
                ood_end = vid_length
            elif match_val == 3:
                ood_start = int(0.25 * vid_length)
                ood_end = int(0.75 * vid_length)
        print('OOD_start: {}'.format(ood_start))
        print('OOD_end: {}'.format(ood_end))
        ood_part = ''
        match = re.findall('_(\D+)(\d+\.\d+)', video)
        print(match)
        if match:
            for m in match:
                if m[0] in self.partition_map and m[1] != '0.0':
                    print('Set OOD part')
                    ood_part = m[0]
        cnt = 0
        while reader.isOpened():
            timing = {}
            ret, frame = reader.read()
            if not ret:
                break
            timing['start'] = time.time()
            frame = self.preprocess(frame)
            timing['preproc'] = time.time()
            dkl = self.inference(frame)
            timing['inference'] = time.time()
            p_i = self.icp(dkl)
            m = self.martingale()
            timing['martingale'] = time.time()
            self.cumsum(m)
            timing['end'] = time.time()
            self.timing_df = self.timing_df.append(timing, ignore_index=True)
            for i, part in enumerate(self.partition_map):
                ground_truth = 0
                if cnt >= ood_start and cnt <= ood_end and part == ood_part:
                    ground_truth = 1
                detection = {
                    'p_i': numpy.squeeze(p_i[i]),
                    'm': numpy.squeeze(m[i]),
                    'cumsum': numpy.squeeze(self.csum[i]),
                    'gt': ground_truth
                }
                self.detect_dfs[part] = self.detect_dfs[part].append(
                    detection,
                    ignore_index=True)
            print('Frame: {} / {}'.format(cnt, vid_length), end = "\r" )
            cnt += 1
            # if cnt ==50:
            #     break
        print('Frame: {} / {}'.format(cnt, vid_length))

    def preprocess(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Perform preprocing on an input frame.  This includes color space
        conversion and resize operations.
        
        Args:
            frame - input image taken from video.
        
        Returns:
            The preprocessed video frame.
        """
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, self.input_d)
        if self.n_chan == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def inference(self, frame: numpy.ndarray) -> numpy.ndarray:
        """Make an inference on the encoder.
        
        Args:
            frame - raw input data to encoder
            
        Returns:
            A (1 x n_latent) vector of the frame's KL-divergence for each
            latent dimension.
        """
        with torch.no_grad():
            frame = torchvision.transforms.functional.to_tensor(frame)
            mu, logvar = self.encoder(frame.unsqueeze(0).to(self.device))
            mu = mu.detach().cpu().numpy()
            logvar = logvar.detach().cpu().numpy()
            return 0.5 * (numpy.power(mu, 2) + numpy.exp(logvar) - logvar - 1)

    def icp(self, dkls: numpy.ndarray) -> numpy.ndarray:
        """Find the percentage of samples in the calibration set with a higher
        KL-divergence than the input sample (Inductive Conformal Prediction).
        
        Args:
            dkls - a (1 x n_latent) vector of KL-divergence for each latent
                dimension.
        
        Returns:
            A (n_partition x 1) vector where entry p_i is the percentage of
            samples in the calibration set with a higher KL-divergence than
            the input for the i-th data partion.
        """
        p_i = numpy.zeros((len(self.partition_map), 1))
        for partition in range(len(self.partition_map)):
            dkl = numpy.sum(dkls * self.z_mask[partition, :])
            p_i[partition, 0] = max(
                numpy.count_nonzero(self.alpha_cal[partition, :] > dkl),
                2) / self.alpha_cal[partition, :].size
        self.past_dkls[:, self.ptr] = numpy.squeeze(p_i)
        self.ptr = (self.ptr + 1) % self.window
        return p_i
    
    def martingale(self) -> numpy.ndarray:
        """Compute the martingale on the detectors ICP output window.
        
        Returns:
            A (1 x n_partition) vector of martingale values for all samples
            currently stored in this classes ICP output window.
        """
        m = numpy.zeros((len(self.partition_map), 1))
        for partition in range(len(self.partition_map)):
            # TODO: try with scikit integral and compare
            for i in range(100):
                m[partition, 0] += numpy.product(
                    self.eps[i] * numpy.power(
                        self.past_dkls[partition, :], self.eps[i] - 1))
        return m

    def cumsum(self, m: float) -> None:
        """Given a new sample, update this class's cumulative summation, which
        is used as the current OOD score.  The cumulative summation can be
        accessed at any time to check the status of a frame as OOD.
        
        Args:
            m - the Martingale resulting from an input frame."""
        m = numpy.nan_to_num(m)
        self.csum = numpy.maximum(0, self.csum + numpy.log(m) - self.decay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test a BetaVAE model on a video.')
    parser.add_argument(
        '--weights',
        help='Path to weights file (encoder weights only).')
    parser.add_argument(
        '--alpha_cal',
        help='Path to alpha calibration json file.')
    parser.add_argument(
        '--grayscale',
        action='store_true',
        help='Video frames are converted to grayscale before inference.')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create a visualization while processing the scenes.')
    parser.add_argument(
        '--video',
        help='Path to input video on which to detect OOD frames.')
    parser.add_argument(
        '--window',
        default=20,
        type=int,
        help='Window size for martingale.')
    parser.add_argument(
        '--decay',
        default=1.0,
        type=float,
        help='Decay term for cummulative summation.')
    args = parser.parse_args()

    runner = BetaVAEDetector(
        args.weights,
        args.alpha_cal,
        int(args.window),
        float(args.decay))
    runner.run_detect(args.video)

    weights_p = list(os.path.split(args.weights))
    weights_p[-1] = weights_p[-1].replace('.pt', '')
    video_p = list(os.path.split(args.video))
    video_p[-1] = video_p[-1].replace('.avi', '')
    output_file = f'{weights_p[-1]}_{video_p[-1]}_window{runner.window}' \
                  f'_decay{runner.decay}.xlsx'
    with pandas.ExcelWriter(output_file) as writer:
        runner.timing_df.to_excel(writer, sheet_name='Times')
        for partition, data_frame in runner.detect_dfs.items():
            data_frame.to_excel(
                writer,
                sheet_name=f'Partition={partition}')
