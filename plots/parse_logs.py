#!/usr/bin/env python3
"""Parse raw console logs from the Duckiebot into a CSV file of event times."""


import argparse
import json
import logging
import re
import pandas as pd


# Use argparser to handle command line arguments in case we need to add more
# options later.
argparser = argparse.ArgumentParser(
    description='Parse JSON file of raw run logs into CSV format.')
argparser.add_argument(
    'file_name',
    help='JSON log dump to parse.')
args = argparser.parse_args()
logging.basicConfig(level=logging.INFO)


# Read in raw data.
logging.info('Reading JSON data...')
raw_data = {}
with open(args.file_name, 'r') as logfp:
    raw_data = json.loads(logfp.read())


# Write times to pandas data frame.
logging.info('Writing times into Pandas data frame...')
parsed_data = pd.DataFrame(
    columns=[
        'Run Name',
        'Motor Stop Time (s)',
        'OOD Detector Start Time (s)',
        'OOD Detector End Time (s)',
        'Image Capture Time (s)',
        'OOD Score'])
detector_regex = re.compile(
    r'\[INFO\]\s+\[(?P<detect_end_stamp>\d+\.\d+)\]:\s+OOD\sFinished:\svalue='
    r'(?P<ood_score>\d+\.\d+)\sim_time=(?P<img_stamp>\d+\.\d+)')
motor_regex = re.compile(
    r'\[INFO\]\s+\[(?P<stop_time>\d+\.\d+)\]:\s+Emergency\sStop\sActivated\!')
idx = 0
for run in raw_data:
    for iter, line in enumerate(raw_data[run].split('\n')):
        result = detector_regex.match(line)
        if result:
            data = result.groupdict()
            parsed_data = parsed_data.append(
                {
                    'Run Name': run,
                    'Motor Stop Time (s)': 0.0,
                    'OOD Detector Start Time (s)': 0.0,
                    'OOD Detector End Time (s)': data['detect_end_stamp'],
                    'Image Capture Time (s)': data['img_stamp'],
                    'OOD Score': data['ood_score']
                },
                ignore_index=True)
            idx += 1
            continue
        result = motor_regex.match(line)
        if result:
            data = result.groupdict()
            parsed_data.loc[idx - 1, 'Motor Stop Time (s)'] = data['stop_time']
            break


# Take care of fields that have inter-row dependencies
logging.info('Calculating remaining fields..')
for run_name in raw_data:
    run_data = parsed_data[parsed_data['Run Name'] == run_name]
    for idx, row in run_data[1:].iterrows():
        parsed_data.loc[idx, 'OOD Detector Start Time (s)'] = \
            run_data.loc[idx - 1, 'OOD Detector End Time (s)']


# Write data to CSV
parsed_data.to_csv('log_data.csv', index=False)
logging.info('Writing to CSV complete.')
