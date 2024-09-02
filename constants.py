
#---------------------------------------------------------------------------
# LOCAL ENVIRONMENT
#---------------------------------------------------------------------------

# TRAIN_DATAPATH="/Users/aditya/Desktop/FYP/project/project_data/"
TRAIN_DATAPATH="/Users/aditya/Large Files/NEW FYP/Train11"

# TEST_DATAPATH="/Users/aditya/Desktop/FYP/project/data/Train4/test"
TEST_DAATPATH="/Users/aditya/Large Files/NEW FYP/Test11"

WEIGHTS_FILE_PATH="/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/pruned_models/base_4_epochs_100/base_4_epoch_100_prune_0.9_.pt"

#---------------------------------------------------------------------------
# GPU ENVIRONMENT
#---------------------------------------------------------------------------

# TRAIN_DATAPATH="/Users/aditya/Desktop/FYP/project/data/train_temp"
# TEST_DATAPATH="/Users/aditya/Desktop/FYP/project/data/test_temp"

############################################################
# WEIGHTS FILES
############################################################


base_model_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/base_model/base_4_bvae_n30_b1.4_ch3_224x224.pt'
base_enc_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/encoder_only/base_4_bvae_n30_b1.4_ch3_224x224.pt'

dq_model_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/quantized_models/dynamic_quant.pt'
dq_enc_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/encoder_only/enc_dynamic_quant.pt'

sq_model_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/quantized_models/static_quant.pt'
sq_enc_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/encoder_only/enc_static_quant.pt'


qat_model_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/quantized_models/qat_quant.pt'
qat_enc_weights = '/Users/aditya/Desktop/FYP/project/code/fyp/models/encoder_only/enc_qat.pt'

############################################################
# CALIB FILES
############################################################

test_folder = '/Users/aditya/Desktop/FYP/project/project_data/Test9/'

base_cal_file = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/calib_files/alpha_cal_base_4_bvae_n30_b1.4_ch3_224x224.json' 
dq_model_cal_file = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/calib_files/dq_model_base_4_bvae_n30_b1.4_ch3_224x224.json'
sq_model_cal_file = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/calib_files/sq_model_static_quant.json'
qat_model_cal_file = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/calib_files/qat_quant.json'

############################################################
# RESULT FILES
############################################################

results_base_model = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/result_xls/base_model/'
results_dq_model = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/result_xls/dq_model/'
results_sq_model = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/result_xls/sq_model/'
results_qat_model = '/Users/aditya/Desktop/FYP/project/code/fyp/out_files/result_xls/qat_model/'

BETA=1.4
N_LATENT=30
BATCH=2
GRAYSCALE=False
N_CHANNELS=3

# SIZES=("224x224" "112x112" "56x56" "28x28" "14x14" "7x7")
INPUT_DIMENSIONS=(224, 224)
EARLY_STOP_PATIENCE = 50


############################################################
# KNOWLEDGE DISTILLATION FILES
############################################################

student_2_all_weights = "/Users/aditya/Desktop/Training_Results/model_files/student_2/student_model2_bvae_n30_b1.4_ch3_224x224.pt"
student_2_enc_wights = "/Users/aditya/Desktop/Training_Results/model_files/student_2/enc/enc_only_student2_bvae_n30_b1.4_ch3_224x224.pt"
student_cal_file = "/Users/aditya/Desktop/Training_Results/calib_files/alpha_cal_student_model2_bvae_n30_b1.4_ch3_224x224.json"
student_2_results_dir = "/Users/aditya/Desktop/Training_Results/OOD_results/student_2/"

student_1_all_weights = "/Users/aditya/Desktop/Training_Results/model_files/student_1/student_model_1_bvae_n30_b1.4_ch3_224x224.pt"
student_1_enc_wights = "/Users/aditya/Desktop/Training_Results/model_files/student_1/enc/enc_only_student_1_bvae_n30_b1.4_ch3_224x224.pt"
student_1_cal_file = "/Users/aditya/Desktop/Training_Results/calib_files/alpha_cal_student_model_1_bvae_n30_b1.4_ch3_224x224.json"
student_1_results_dir = "/Users/aditya/Desktop/Training_Results/OOD_results/student_1/"