import torch
import copy
import time
import os 
import json
import pandas
import re

from quantization.quant_bvae import BetaVae,Encoder
from knowledge_distilation.student_arch_1 import ModBetaVae, ModEncoder
from utils.data_loader import get_data_loader
from ood_detection.calibration import get_partition_variance
from ood_detection.find_optimal_decay import optimize_decay, get_roc
from ood_detection.test import BetaVAEDetector
from constants import *

from quantization.dynamic_quantisation import dynamic_quantise
from quantization.static_quantisation import static_quantise
from quantization.quantization_aware_training import qat_quantise

from knowledge_distilation.student_arch_2 import Mod2BetaVae, Mod2Encoder


############################
# Get Encoder only Model
############################
def get_encoder_only_model(weight_file, device):
    
    full_model=ModBetaVae(
        N_LATENT,
        BETA,
        N_CHANNELS,
        INPUT_DIMENSIONS,
        BATCH)

    weights = weight_file
    # full_model = qat_quantise(full_model)
    full_model.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))

    encoder = ModEncoder(
        n_latent=N_LATENT,
        n_chan=N_CHANNELS,
        input_d=INPUT_DIMENSIONS)

    # encoder = qat_quantise(encoder)

    encoder_dict = encoder.state_dict()
    full_model_dict = full_model.state_dict()

    for key in encoder_dict:
        encoder_dict[key] = full_model_dict[key]

    torch.save(encoder_dict, f'/Users/aditya/Desktop/Training_Results/model_files/student_1/enc/enc_only_student_1_bvae_n30_b1.4_ch3_224x224.pt')


#####################################################################
# Calibrate model and determine post-processing hyperparameters.
#####################################################################

def calibrate_model(calib_dataset, calib_folder, weight_file, device, quantize = None):
    model=ModBetaVae(
        N_LATENT,
        BETA,
        N_CHANNELS,
        INPUT_DIMENSIONS,
        1)

    if quantize == 'dq':
        model = dynamic_quantise(model)

    elif quantize == 'sq':
        print('A')
        model = static_quantise(model)

    elif quantize == 'qat':
        model = qat_quantise(model)

    model.load_state_dict(copy.deepcopy(torch.load(weight_file, device)))

    model.eval()
    # model.to(device)
    alpha_cal = {}
    alpha_cal['PARAMS'] = {}
    alpha_cal['PARAMS']['n_latent'] = model.n_latent
    alpha_cal['PARAMS']['input_d'] = model.input_d
    alpha_cal['PARAMS']['n_chan'] = model.n_chan

    for partition in os.listdir(calib_dataset):
        print(f'Processing partition: {partition}')
        alpha_cal[partition] = get_partition_variance(
            os.path.join(calib_dataset, partition),
            model)
        print(f'Rankings for partition: {partition}')
        for rank, value in enumerate(alpha_cal[partition]['top_z']):
            print(f'{rank}: {value}')

    dest_path = list(os.path.split(weight_file))

    if quantize == 'dq':
        file_name = f'dq_model_{dest_path[-1].replace("pt", "json")}'
    
    elif quantize == 'sq':
        file_name = f'sq_model_{dest_path[-1].replace("pt", "json")}'
    
    elif quantize == 'qat':
        file_name = f'qat_model_{dest_path[-1].replace("pt", "json")}'
    
    else:
        file_name = f'alpha_cal_{dest_path[-1].replace("pt", "json")}'     
    
    with open(calib_folder+file_name, 'w') as alpha_cal_f:
        alpha_cal_f.write(json.dumps(alpha_cal))

#####################################################################
# Find optimal decay term for CUMSUM.
#####################################################################

def optimum_decay(detection_results, cal, data_partition):

    partition_data = {}
    for file in detection_results:
        with pandas.ExcelFile(file) as xls_f:
            sheets = pandas.read_excel(xls_f, None)
            for sheet in sheets:
                match = re.match('Partition=(\D+)', sheet)
                if match:
                    partition = match.groups()[0]
                    if partition not in partition_data:
                        partition_data[partition] = []
                    partition_data[partition].append(sheets[sheet])
    fpr, tpr, opt_decay, opt_auroc = optimize_decay(partition_data[data_partition])
    print(
        f'Optimal decay term for partition {data_partition} is {opt_decay} '
        f'with an AUROC of {opt_auroc}.')
    print(
        f'FPR: {fpr}, TPR: {tpr}')
    
    cal_data = {}
    with open(cal, 'r') as cal_f:
        cal_data = json.loads(cal_f.read())
    if 'decay' not in cal_data['PARAMS']:
        cal_data['PARAMS']['decay'] = {}
    cal_data['PARAMS']['decay'][partition] = opt_decay
    with open(cal, 'w') as cal_f:
        cal_f.write(json.dumps(cal_data))

#####################################################################
# Run OOD
#####################################################################

def run_ood(window, decay, weights, video, alpha_cal, output, quantize = None):
    runner = BetaVAEDetector(
        weights,
        alpha_cal,
        int(window),
        float(decay), quantize)

    runner.run_detect(video)

    weights_p = list(os.path.split(weights))
    weights_p[-1] = weights_p[-1].replace('.pt', '')
    video_p = list(os.path.split(video))
    video_p[-1] = video_p[-1].replace('.avi', '')
    output_file = f'{weights_p[-1]}_{video_p[-1]}_window{runner.window}' \
                    f'_decay{runner.decay}.xlsx'

    with pandas.ExcelWriter(output + output_file) as writer:
        runner.timing_df.to_excel(writer, sheet_name='Times')
        for partition, data_frame in runner.detect_dfs.items():
            data_frame.to_excel(
                writer,
                sheet_name=f'Partition={partition}')

#####################################################################
# GET OOD Metric for a given decay value
#####################################################################

def ood_metrics(detection_results, data_partition, decay_value):

    partition_data = {}
    for file in detection_results:
        with pandas.ExcelFile(file) as xls_f:
            sheets = pandas.read_excel(xls_f, None)
            for sheet in sheets:
                match = re.match('Partition=(\D+)', sheet)
                if match:
                    partition = match.groups()[0]
                    if partition not in partition_data:
                        partition_data[partition] = []
                    partition_data[partition].append(sheets[sheet])

    fpr, tpr, thresholds, auroc = get_roc(partition_data[data_partition], decay_value)

    print('fpr:{}\n'
          'tpr:{}\n'.format(fpr, tpr))
        #   'thresholds:{}\n'.format(fpr, tpr, thresholds))

    print('AUROC: {}'.format(auroc))

#####################################################################
# Function calls
#####################################################################
train_04_all_weights = "/Users/aditya/Desktop/Training_Results/model_files/encoder_only/enc_only_train0_4_bvae_n30_b1.4_ch3_224x224.pt"
train_59_all_weights = "/Users/aditya/Desktop/Training_Results/model_files/59/train5_9_bvae_n30_b1.4_ch3_224x224.pt"

train_04_enc_weights = "/Users/aditya/Desktop/Training_Results/model_files/encoder_only/enc_only_train0_4_bvae_n30_b1.4_ch3_224x224.pt" 
train_59_enc_weights = "/Users/aditya/Desktop/Training_Results/model_files/59/enc/enc_only_train_59_bvae_n30_b1.4_ch3_224x224.pt"

train_59_cal_file = "/Users/aditya/Desktop/Training_Results/calib_files/alpha_cal_train5_9_bvae_n30_b1.4_ch3_224x224.json"
train_04_cal_file = "/Users/aditya/Desktop/Training_Results/calib_files/alpha_cal_train0_4_bvae_n30_b1.4_ch3_224x224.json"

train_04_results_dir = "/Users/aditya/Desktop/Training_Results/OOD_results/train_04/"
train_59_results_dir = "/Users/aditya/Desktop/Training_Results/OOD_results/train_59/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

#####################################################################
# 1. Get encoder only model

weights = student_1_all_weights
# get_encoder_only_model(weights, device)

#####################################################################
# 2. Calibrate model

calib_folder = '/Users/aditya/Desktop/Training_Results/calib_files/'
calib_dataset = '/Users/aditya/Desktop/FYP/project/project_data/Calibration9'

# weights = qat_model_weights
# calibrate_model(calib_dataset, calib_folder, weights, device, quantize=None)

#####################################################################
# 3. Test OOD detection

weights = student_1_enc_wights

# for test_vid in os.listdir(test_folder):
#     s = time.time()
#     print()
#     print('Testing file: {}'.format(test_vid))
#     run_ood(window=20, decay=22, weights=weights, video=test_folder+test_vid, alpha_cal=student_1_cal_file, output=student_1_results_dir, quantize=None)
#     print("Time Spent: {}".format(time.time() - s))
#     print('------------------------------------------')
    

#####################################################################
# 4. Find optimum decay term
results_files = []
for result_xl in os.listdir(student_1_results_dir):
    results_files.append(student_1_results_dir+'/'+result_xl)

# optimum_decay(results_files, student_1_cal_file, data_partition='brightness')

#####################################################################
# 5. Get OOD metrics with optimal decay
opt_decay = 1.701701701701
ood_metrics(results_files, data_partition='brightness', decay_value=opt_decay)




