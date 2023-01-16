#!/usr/bin/env python3
"""Plot the response times for the Duckiebot with various task
combinations."""


from typing import Tuple
import re


from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def get_times(f: str) -> Tuple[np.ndarray]:
    """Get a list of YOLO and OOD detector response times from a log file.

    Args:
        f - filename to read

    Returns:
        Tuple ([OOD detection times], [YOLO detection times])
    """
    ood_times = []
    yolo_times = []
    with open(f, 'r') as file_obj:
        ood_state = 'wait'
        yolo_state = 'wait'
        for line in file_obj:
            if yolo_state == 'wait':
                if 'YOLO: Recieved an image...' in line:
                    yolo_r = float(re.search(r'\[(\d+\.\d+)\]', line).group(1))
                    yolo_state = 'process'
            elif yolo_state == 'process':
                if 'YOLO: Done...' in line:
                    done = float(re.search(r'\[(\d+\.\d+)\]', line).group(1))
                    yolo_times.append(done - yolo_r)
                    yolo_state = 'wait'
            if ood_state == 'wait':
                if 'OOD Detector: Recieved an image...' in line:
                    ood_r = float(re.search(r'\[(\d+\.\d+)\]', line).group(1))
                    ood_state = 'process'
            elif ood_state == 'process':
                if 'OOD Detector: Got OOD score:' in line:
                    done = float(re.search(r'\[(\d+\.\d+)\]', line).group(1))
                    ood_times.append(done - ood_r)
                    ood_state = 'wait'
    return 1000*np.array(ood_times), 1000*np.array(yolo_times)


def get_times_for_lf(f: str) -> np.ndarray:
    """Get the response times for Duckietown's in-built lane following from a
    log file.

    Args:
        f - the log file to parse

    Returns:
        ndarray of response times in ms.
    """
    times = []
    with open(f, 'r') as file_obj:
        curr_time = 0
        for line in file_obj:
            if ' secs:' in line:
                #print(line.split()[1])
                curr_time = float(line.split()[1])
            elif 'nsecs:' in line:
                curr_time += float(line.split()[1]) * 1e-9
                times.append(curr_time)
                curr_time = 0
    deltas = []
    for i in range(len(times) - 1):
        delta = times[i + 1] - times[i]
        #if delta > 0:
        deltas.append(delta)
    return 1000 * np.array(deltas)


if __name__ == '__main__':
    # Read in the data from log files
    lf_iso = get_times_for_lf('iso_lf.log')
    ood_t1, yolo_t1 = get_times('yolo_ood.out')
    ood_t2, yolo_t2 = get_times('yolo_lf.out')
    ood_t3, yolo_t3 = get_times('ood_lf.out')
    ood_t4, yolo_t4 = get_times('yolo_ood_lf.out')
    lf_t2 = get_times_for_lf('yolo_lf.log')
    lf_t3 = get_times_for_lf('ood_lf.log')
    lf_t4 = get_times_for_lf('yolo_ood_lf.log')

    #Plot the data
    fig, ax = plt.subplots()
    p1 = ax.violinplot([ood_t1, [-100], ood_t3, ood_t4], vert=False)
    p2 = ax.violinplot([yolo_t1, yolo_t2, [-100], yolo_t4], vert=False)
    p3 = ax.violinplot([[-100], lf_t2, lf_t3, lf_t4], vert=False)
    ax.legend(
        [
            mpatches.Patch(color=p1["bodies"][0].get_facecolor().flatten()), mpatches.Patch(color=p2["bodies"][0].get_facecolor().flatten()),
            mpatches.Patch(color=p3["bodies"][0].get_facecolor().flatten())
        ],
        ['OOD Detection (A)', 'Object Detection (B)', 'Lane Following (C)'],
        loc='upper center',
        bbox_to_anchor=(0.5, -0.1),
        ncol=3
    )
    ax.set_xlabel('Response Time (ms)')
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['Tasks A, B', 'Tasks B, C', 'Tasks A, C', 'Tasks A, B, C'])
    ax.set_xlim([0, 4750])
    plt.show()

