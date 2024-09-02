import torch
import copy

from bvae import BetaVae
from knowledge_distilation.student_arch_1 import ModBetaVae
from knowledge_distilation.student_arch_2 import Mod2BetaVae
from knowledge_distilation.train_student import train_student_model
from utils.data_loader import get_data_loader
from constants import * 
from torchsummary import summary

print('________________________________________________________________')
print(f'Starting student training for input size {INPUT_DIMENSIONS}')
print(f'beta={BETA}')
print(f'n_latent={N_LATENT}')
print(f'batch={BATCH}')
print(f'Using data set {TRAIN_DATAPATH}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, _ = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, 0.8)

print("Train Set: {}".format(len(train_loader)))
print("Val Set: {}".format(len(val_loader)))

teacher_model=BetaVae(
    N_LATENT,
    BETA,
    N_CHANNELS,
    INPUT_DIMENSIONS,
    BATCH)

weights = "/Users/aditya/Desktop/FYP/project/code/fyp/models/base_model/base_4_bvae_n30_b1.4_ch3_224x224.pt"
teacher_model.load_state_dict(copy.deepcopy(torch.load(weights, device)))
    
student_model = ModBetaVae(
    N_LATENT,
    BETA,
    N_CHANNELS,
    INPUT_DIMENSIONS,
    2#BATCH
)

# train_student_model(
#     teacher_model,
#     student_model,
#     weights_file='/Users/aditya/Desktop/Training_Results/model_files/'+f'student_model_bvae_n{N_LATENT}_b{BETA}_ch{N_CHANNELS}_'
#                     f'{"x".join([str(i) for i in INPUT_DIMENSIONS])}.pt',
#     train_loader=train_loader, 
#     val_loader=val_loader,
#     epochs=1,
#     lr=1e-4,
#     distil_weight=0.7,
#     file_prefix='student_model'
#     )


summary(student_model, (3, 224, 224))

