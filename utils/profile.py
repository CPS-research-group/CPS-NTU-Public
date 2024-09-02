from torch.autograd import profiler
import torch
from utils.data_loader import get_data_loader

from quantization.quant_bvae import BetaVae 
from constants import *

from quantization.static_quantisation import static_quantise
from quantization.dynamic_quantisation import dynamic_quantise
from quantization.quantization_aware_training import qat_quantise

model=BetaVae(
        N_LATENT,
        BETA,
        N_CHANNELS,
        INPUT_DIMENSIONS,
        BATCH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = base_model_weights
# model = qat_quantise(model)
model.to(device)
model.load_state_dict(torch.load(weights,map_location=torch.device('cpu')))


# weights_file = '/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/base_models/base_4_model_100_epochs/base_4_bvae_n30_b1.4_ch3_224x224.pt'
# weights_file = WEIGHTS_FILE_PATH


def profile_model(model, input, rows=10, cuda=False):
    with profiler.profile(profile_memory=False, record_shapes=True, use_cuda=cuda) as prof:
        with profiler.record_function("model_inference"):
            model(input)
        
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
    # return str(prof.key_averages().table(
        # sort_by="cpu_time_total", row_limit=rows))


# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

# input = None
_, _, test_loader = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, 0.8)

for i, data in enumerate(test_loader):
    input, _ = data
    input = input.to(device)
    break 

# print(profile_model(model, input))
profile_model(model, input)