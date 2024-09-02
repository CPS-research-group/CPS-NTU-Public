#!/usr/bin/env python3
"""Find the optimal decay term for the cumsum, given a series of test CV data."""


from typing import List, Tuple
import argparse
import json
import re
import warnings
import matplotlib.pyplot
import numpy
import pandas
import sklearn.metrics


warnings.filterwarnings('ignore')


def get_roc(data: List[pandas.DataFrame],
            decay: float) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    """Given a list of evalutated scenes with Martingale scores and their
    corresponding ground truth values, calculate the FPR, TPR, and AUROC for a
    given decay value.
    
    Args:
        data - list of Pandas DataFrames, each corresponding to a separate
            video and containing at least a column 'm' of Martingale scores
            and a column 'gt' of corresponding ground truth values.
        decay - decay term in the cummulative sum.
        
    Returns:
        A tuple (FPR, TPR, AUROC) for all DataFrames analyzed.
    """
    gt_all = numpy.array([])
    csum_all = numpy.array([])
    for sheet in data:
        m = sheet['m'].to_numpy()
        gt = sheet['gt'].to_numpy()
        csum = numpy.zeros(m.shape)
        for idx in range(1,m.size):
            csum[idx] = max(
                0,
                csum[idx - 1] + numpy.log(numpy.nan_to_num(m[idx])) - decay)
        gt_all = numpy.append(gt_all, gt)
        csum_all = numpy.append(csum_all, csum)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt_all, csum_all)
    auroc = sklearn.metrics.roc_auc_score(gt_all, csum_all)
    return fpr, tpr, thresholds, auroc


def optimize_decay(partition_data: List[pandas.DataFrame]) -> Tuple[float]:
    """Given Martingale scores for several scenes in a partition, find the
    optimal decay term for the cummulative sum algorithm.
    
    Args:
        data - list of Pandas DataFrames, each corresponding to a separate
            video and containing at least a column 'm' of Martingale scores
            and a column 'gt' of corresponding ground truth values.
    
    Returns:
        Tuple (optimimal decay term, AUROC for optimal decay term)
    """
    decay_sweep = numpy.linspace(0, 100, 1000)
    top_ten = [([], [], 0, 0)] * 10
    for i, decay in enumerate(list(decay_sweep)):
        print('Testing decay: [{} / {}]: {}'.format(i, len(decay_sweep), decay), end = "\r" )
        fpr, tpr, _, auroc = get_roc(partition_data, decay)
        for idx, place in enumerate(top_ten):
            if auroc > place[3]:
                top_ten.insert(idx, (fpr, tpr, decay, auroc))
                top_ten = top_ten[:10]
                break
    print('Testing decay: [{} / {}]: {}'.format(i, len(decay_sweep), decay))

    for place in top_ten:
        matplotlib.pyplot.plot(
            place[0],
            place[1],
            label=f'Decay={place[2]}, AUROC={place[3]}')
    matplotlib.pyplot.legend(loc=4)
    matplotlib.pyplot.show()
    return top_ten[0]#[2:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Find the optimal decay term for cumsum in a BetaVAE OOD detector.')
    parser.add_argument(
        'detection_results',
        nargs='+',
        help='All detection results files to be considered when determining '
             'the optimal decay term.')
    parser.add_argument(
        '--cal',
        help='Path to calibration file to update after finding optimal term')
    parser.add_argument(
        '--partition',
        help='Name of partition to determine decay term for.')
    args = parser.parse_args()

    partition_data = {}
    for file in args.detection_results:
        with pandas.ExcelFile(file) as xls_f:
            sheets = pandas.read_excel(xls_f, None)
            for sheet in sheets:
                match = re.match('Partition=(\D+)', sheet)
                if match:
                    partition = match.groups()[0]
                    if partition not in partition_data:
                        partition_data[partition] = []
                    partition_data[partition].append(sheets[sheet])
    opt_decay, opt_auroc = optimize_decay(partition_data[args.partition])
    print(
        f'Optimal decay term for partition {args.partition} is {opt_decay} '
        f'with an AUROC of {opt_auroc}.')
    cal_data = {}
    with open(args.cal, 'r') as cal_f:
        cal_data = json.loads(cal_f.read())
    if 'decay' not in cal_data['PARAMS']:
        cal_data['PARAMS']['decay'] = {}
    cal_data['PARAMS']['decay'][partition] = opt_decay
    with open(args.cal, 'w') as cal_f:
        cal_f.write(json.dumps(cal_data))
    """
    for partition in partition_data:
        if args.partition == partition:
            opt_decay, opt_auroc = optimize_decay(partition_data[partition])
            print(
                f'Optimal decay term for partition {partition} is {opt_decay} '
                f'with an AUROC of {opt_auroc}.')
            cal_data = {}
            with open(args.cal, 'r') as cal_f:
                cal_data = json.loads(cal_f.read())
            if 'decay' not in cal_data['PARAMS']:
                cal_data['PARAMS']['decay'] = {}
            cal_data['PARAMS']['decay'][partition] = opt_decay
            with open(args.cal, 'w') as cal_f:
                cal_f.write(json.dumps(cal_data))
    """
