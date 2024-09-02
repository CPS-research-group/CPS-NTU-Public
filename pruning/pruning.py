import copy
from ntpath import join
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os
# from torchsummary import summary

from quantization.quant_bvae import BetaVae 
from constants import *   

def get_module_sparsity(module):
    return 100. * float(torch.sum(module.weight == 0)) / float(module.weight.nelement())
    
def get_global_sparsity(params_to_prune):

    total_sum = torch.tensor(0)
    total_nelement = 0

    for param in params_to_prune:
        
        # print('{} : {}'.format(type(param[0]).__name__,get_module_sparsity(param[0])))
        # print(get_module_sparsity(param[0]))
        total_sum += torch.sum(param[0].weight == 0) 
        total_nelement += param[0].weight.nelement()

    sparsity = 100. * float(total_sum) / float(total_nelement)
    return sparsity

def permanent_prune(params_to_prune):
    for param in params_to_prune:
        prune.remove(param[0], 'weight')

    
######################################################
## PRUNING METHODS
######################################################

def random_pruing(module):

    print("Sparsity before pruning : {:.2f}%".format(get_module_sparsity(model)))
    prune.random_unstructured(module, name="weight", amount=0.3)
    print("Sparsity After pruning : {:.2f}%".format(get_module_sparsity(model)))


def global_pruning(modules_to_prune, amount):

    print("Sparsity before pruning : {:.2f}%".format(get_global_sparsity(modules_to_prune)))
    prune.global_unstructured(
        modules_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

def layerwise_pruning(module, amount):
    #dim = 0 -> prune channels
    #dim = 1 -> prune filters
    #dim = 2 -> prune weights
    #dim = 3 -> prune biases

    prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)


def experiment_multiple_prune_amounts():
    for i in range(1, 10):

        global_pruning(parameters_to_prune, i/10)#0.9+i/100)
        print("Sparsity After pruning : {:.2f}%".format(get_global_sparsity(parameters_to_prune)))
        permanent_prune(parameters_to_prune)

        # print(list(model.enc_dense1.named_parameters()))
        # print(list(model.named_parameters()))
        # print(list(model.enc_dense1.named_parameters()))
        # torch.save(model.state_dict(), 'experiments/pruned_models/base_4_epochs_100/base_4_epoch_100_prune_{}_.pt'.format(i/10))#0.9+i/100))


def experiment_layerwise_pruning():

    for i in range(1, 10):
        layerwise_pruning(model.enc_conv1, i/10)
        layerwise_pruning(model.enc_conv2, i/10)
        layerwise_pruning(model.enc_conv3, i/10)
        layerwise_pruning(model.enc_conv4, i/10)

        layerwise_pruning(model.enc_dense1, i/10)
        layerwise_pruning(model.enc_dense2, i/10)
        layerwise_pruning(model.enc_dense3, i/10)

        layerwise_pruning(model.dec_dense4, i/10)
        layerwise_pruning(model.dec_dense3, i/10)
        layerwise_pruning(model.dec_dense2, i/10)
        layerwise_pruning(model.dec_dense1, i/10)

        layerwise_pruning(model.dec_conv4, i/10)
        layerwise_pruning(model.dec_conv3, i/10)
        layerwise_pruning(model.dec_conv2, i/10)
        layerwise_pruning(model.dec_conv1, i/10)

        print("Sparsity After pruning : {:.2f}%".format(get_global_sparsity(parameters_to_prune)))
        permanent_prune(parameters_to_prune)

        # print(list(model.enc_dense1.named_parameters()))
        # print(list(model.named_parameters()))
        # print(list(model.enc_dense1.named_parameters()))
        # torch.save(model.state_dict(), 'experiments/pruned_models/base_4_epochs_100/base_4_epoch_100_prune_{}_.pt'.format(i/10))#0.9+i/100))


model = BetaVae(N_LATENT, BETA, N_CHANNELS,
    input_d=INPUT_DIMENSIONS,
    batch=1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_file = '/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/base_models/base_4_model_100_epochs/base_4_bvae_n30_b1.4_ch3_224x224.pt'
model.load_state_dict(copy.deepcopy(torch.load(weights_file, device)))

parameters_to_prune = (
    (model.enc_conv1, 'weight'),
    (model.enc_conv2, 'weight'),
    (model.enc_conv3, 'weight'),
    (model.enc_conv4, 'weight'),

    (model.enc_dense1, 'weight'),
    (model.enc_dense2, 'weight'),
    (model.enc_dense3, 'weight'),

    (model.dec_dense4, 'weight'),
    (model.dec_dense3, 'weight'),
    (model.dec_dense2, 'weight'),
    (model.dec_dense1, 'weight'),

    (model.dec_conv4, 'weight'),
    (model.dec_conv3, 'weight'),
    (model.dec_conv2, 'weight'),
    (model.dec_conv1, 'weight')
    )

module_names = ['enc_conv1', 'enc_conv2', 'enc_conv3', 'enc_conv4', 'enc_dense1', 'enc_dense2', 'enc_dense3', 'dec_dense4', 'dec_dense3', 'dec_dense2', 'dec_dense1', 'dec_conv4', 'dec_conv3', 'dec_conv2', 'dec_conv1']

for j, module in enumerate(parameters_to_prune):
    if not os.path.isdir('experiments/layerwise_pruning/{}'.format(module_names[j])):
        os.makedirs('experiments/layerwise_pruning/{}'.format(module_names[j]))

    for i in range(1, 11):
        model.load_state_dict(copy.deepcopy(torch.load(weights_file, device)))
        layerwise_pruning(module[0], i/10)
        print("Sparsity After pruning : {:.2f}%".format(get_global_sparsity([module])))
        permanent_prune([module])
        torch.save(model.state_dict(), 'experiments/layerwise_pruning/{}/layer_{}_{}.pt'.format(module_names[j], module_names[j], i*10))#0.9+i/100))
  

