import torch
import copy
import os

from bvae import BetaVae, Encoder
from quantization.quant_bvae import QuantBetaVae, QuantEncoder
# from utils.data_loader import get_data_loader
from quantization.dynamic_quantisation import dynamic_quantise
# from quantization.static_quantisation import static_quantise
# from quantization.quantization_aware_training import qat_quantise
# import test_model

from constants import *


def get_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')
    return size


if __name__ == '__main__':

    n_latent = N_LATENT
    beta = BETA
    input_dimensions = INPUT_DIMENSIONS
    batch = BATCH
    n_channels = N_CHANNELS
    grayscale = GRAYSCALE

    model = BetaVae(
        n_latent,
        beta,
        n_channels,
        input_dimensions,
        batch)

    qmodel = QuantBetaVae( 
        n_latent,
        beta,
        n_channels,
        input_dimensions,
        batch)

    encoder = Encoder(
        n_latent=N_LATENT,
        n_chan=N_CHANNELS,
        input_d=INPUT_DIMENSIONS)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Complete Model
    weights_file = base_model_weights
    model.load_state_dict(copy.deepcopy(torch.load(weights_file, device)))
    model.eval()

    # Encoder only model
    # weights_file = base_enc_weights 
    # encoder.load_state_dict(copy.deepcopy(torch.load(weights_file, device)))
    # encoder.eval()

    #########################################
    # DYNAMIC QUANTIZATION 
    #########################################
    
    dynamic_quantized_model = dynamic_quantise(model)
    
    #########################################
    # STATIC QUANTIZATION 
    #########################################
    
    # static_quantized_model = static_quantise(qmodel, calibration=True) 

    #########################################
    # QUANTIZATION AWARE TRAINING 
    #########################################
    
    # qat_model = qat_quantise(qmodel, train=True) 

    #########################################
    # quantized_model = qat_model
    # q_model_path = 'quantized_models/quant_bvae.pt'
    # torch.save(quantized_model.state_dict(), q_model_path)

    # Compare Model Size
    orig_size = get_model_size(model)
    quant_size = get_model_size(dynamic_quantized_model)
    print(" Origianal Model: %.2f MB" %(orig_size))
    print(" Quantized Model: %.2f MB" %(quant_size))
    print('{0:.2f} times smaller'.format(orig_size/quant_size))

    ## Test Accuracy 
    # print('-------------------------')
    # print("TESTING QUANTIZED MODEL")
    # with torch.no_grad():
    #     _, _, test_loader = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, 0.8)
    #     avg_ce_loss, avg_kl_loss, total_loss = test_model.test_model(quantized_model, test_loader, BETA)
        

