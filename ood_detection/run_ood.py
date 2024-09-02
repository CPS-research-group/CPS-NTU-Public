from sys import implementation
import pandas
import os

from ood_detection.test import BetaVAEDetector
from constants import * 
import torch

def run_ood(window, decay, weights, video, alpha_cal, device):
    runner = BetaVAEDetector(
        weights,
        alpha_cal,
        int(window),
        float(decay))

    runner.run_detect(video)

    weights_p = list(os.path.split(weights))
    weights_p[-1] = weights_p[-1].replace('.pt', '')
    video_p = list(os.path.split(video))
    video_p[-1] = video_p[-1].replace('.avi', '')
    output_file = f'{weights_p[-1]}_{video_p[-1]}_window{runner.window}' \
                    f'_decay{runner.decay}.xlsx'

    with pandas.ExcelWriter(output_file) as writer:
        runner.timing_df.to_excel(writer, sheet_name='Times')
        for partition, data_frame in runner.detect_dfs.items():
            data_frame.to_excel(
                writer,
                sheet_name=f'Partition={partition}')
