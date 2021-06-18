#!/usr/bin/env python3
"""Generate the plots from the DESTION2021 paper from the raw data."""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


log_data = pd.read_csv('log_data.csv')
measured_data = pd.read_csv('measured_data.csv')


# Figue 6: Distribution of OOD scores as a function of dis-tance from the
# starting position. The scores are based on im-ages that are captured at these
# distances. Each line denotesall of the OOD scores during a specific test run.
# Line stylesare used to classify the various runs based on the achieved
# stopping distance from the obstacle.
runs = pd.DataFrame(
    columns = [
        'Run Name',
        'Distance Travelled',
        'OOD Score',
        'Color',
        'Width',
        'Style',
        'Label'])
for _, run in measured_data.iterrows():
    frame = {}
    run_data = log_data[log_data['Run Name'] == run['Run Name']]
    total_distance = 75.0 - run.at['Distance to Obstacle (cm)']
    total_time = run_data.at[run_data.index[-1], 'Motor Stop Time (s)'] - \
        run_data.at[run_data.index[0], 'Image Capture Time (s)']
    travel_time = run_data['Image Capture Time (s)'] - \
        run_data.at[run_data.index[0], 'Image Capture Time (s)']
    frame = {
        'Distance Traveled': list((total_distance / total_time) * travel_time),
        'OOD Score': list(run_data['OOD Score'])
    }
    if total_distance >= 75:
        frame['Color'] = '#d55e00'
        frame['Width'] = 1.6
        frame['Style'] = 'solid'
        frame['Label'] = 'Collided with Obstacle'
    elif total_distance >= 55:
        frame['Color'] = '#0072b2'
        frame['Width'] = 0.6
        frame['Style'] = 'solid'
        frame['Label'] = '<20cm stopping distance'
    elif total_distance >= 35:
        frame['Color'] = '#0072b2'
        frame['Width'] = 0.6
        frame['Style'] = 'dashed'
        frame['Label'] = '<40cm stopping distance'
    elif total_distance >= 15:
        frame['Color'] = '#0072b2'
        frame['Width'] = 0.6
        frame['Style'] = 'dashdot'
        frame['Label'] = '<60cm stopping distance'
    else:
        frame['Color'] = '#0072b2'
        frame['Width'] = 0.6
        frame['Style'] = 'dotted'
        frame['Label'] = '<80cm stopping distance'
    runs = runs.append(frame, ignore_index=True)
_, ax = plt.subplots()
for _, run in runs.iterrows():
    ax.plot(
        list(run['Distance Traveled']),
        list(run['OOD Score']),
        run['Color'],
        linewidth=run['Width'],
        linestyle=run['Style'],
        label=run['Label'],
        solid_capstyle='round')
ax.hlines(0.11, 0, 75, colors='#009e73', linestyles='solid', linewidth=3.2)
ax.annotate('OOD\nDetection\nThreshold', (63, 0.07))
ax.vlines(15, 0.02, 0.24, colors='#cc79a7', linestyle = 'solid', linewidth=3.2)
ax.annotate('Risk Zone Boundary', (5, 0.01))
ax.vlines(75, 0.025, 0.3, colors='#000000', linestyle='solid', linewidth=3.2)
ax.annotate('Obstacle Location', (59, 0.01))
ax.set_xlabel('Distance from Start Position to Image Capture Location (cm.)')
ax.set_ylabel('OOD Score')
handles, labels = plt.gca().get_legend_handles_labels()
all_hl = sorted(list(zip(labels, handles)), key=lambda x: x[0])
idx = [i for i, j in enumerate(all_hl) if j[0].startswith('C')][0]
all_hl.extend(all_hl[:idx])
condensed_labels = dict(all_hl[idx:])
plt.legend(condensed_labels.values(), condensed_labels.keys(), loc='best')
plt.tight_layout()
plt.show()


# Figure 7: Violin plot of sub-task execution times for all the test runs.
end_to_end_times = (measured_data['End Time (ns)'] - \
    measured_data['Start Time (ns)']) / 1e9
end_to_end_times = end_to_end_times[end_to_end_times > 0]
filtered_log = log_data[log_data['OOD Detector Start Time (s)'] > 0]
detectstart_to_detectend = filtered_log['OOD Detector End Time (s)'] - \
    filtered_log['OOD Detector Start Time (s)']
stops = log_data[log_data['Motor Stop Time (s)'] != 0]
detectend_to_estop = stops['Motor Stop Time (s)'] - \
    stops['OOD Detector End Time (s)']
capture_to_detectstart = filtered_log['OOD Detector Start Time (s)'] - \
    filtered_log['Image Capture Time (s)']
capture_to_detectstart[capture_to_detectstart < 0] = 0
data = [
    list(end_to_end_times),
    list(detectstart_to_detectend),
    list(detectend_to_estop),
    list(capture_to_detectstart)]
_, ax = plt.subplots()
parts = ax.violinplot(data, vert=False, showextrema=False)
colors = ['#e69f00', '#56b4e9', '#009e73', '#f0e442']
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_edgecolor('#000000')
    pc.set_alpha(1)
medians = [np.percentile(ds, 50) for ds in data]
ax.scatter(
    [np.percentile(ds, 50) for ds in data],
    np.arange(1, len(data) + 1),
    marker='o',
    color='#ffffff',
    s=10)
ax.set_title('Timing Distribution Comparison')
ax.set_xlabel('Time (sec.)')
ax.set_yticks(np.arange(1, len(data) + 1))
ax.set_yticklabels([' '] * len(data))
plt.legend(
    [
        'End-to-end stop time',
        'OOD detector execution time',
        'Detection result to motor stop time',
        'Image capture to detection start time'])
plt.show()


# Figure 8: Boxplots with confidence interval showing the distribution of
# projected stopping distances for varying OOD detection thresholds. The
# middle-quartile line marks the estimated median value.
thresholds = [0.01 * i for i in range(7, 12)]
stopping_distance = [[] for i in range(len(thresholds))]
for _, run in measured_data.iterrows():
    run_data = log_data[log_data['Run Name'] == run['Run Name']]
    total_distance = 75.0 - run.at['Distance to Obstacle (cm)']
    total_time = run_data.at[run_data.index[-1], 'Motor Stop Time (s)'] - \
        run_data.at[run_data.index[0], 'Image Capture Time (s)']
    travel_time = run_data['OOD Detector End Time (s)'] - \
        run_data.at[run_data.index[0], 'Image Capture Time (s)']
    dist_data = run_data.copy()
    dist_data['Distance'] = list(
        75 - (total_distance / total_time) * travel_time)
    for idx, th in enumerate(thresholds):
        above_th = dist_data[dist_data['OOD Score'] >= th].head(1)
        stopping_distance[idx].append(
            above_th.at[above_th.index[0], 'Distance'])
_, ax = plt.subplots()
ax.boxplot(stopping_distance, notch=True)
ax.set_title('Simulated Threshold Adjustment vs. Stopping Distance')
ax.set_xlabel('OOD Detection Threshold')
ax.set_xticklabels(thresholds)
ax.set_ylabel('Estimated Stopping Distance from Obstacle (cm.)')
ax.hlines(60, 0.5, 5.5, colors='#cc79a7', linestyle='solid', linewidth=3.2)
ax.annotate('Risk Zone Boundary', (4, 55))
plt.tight_layout()
plt.show()
