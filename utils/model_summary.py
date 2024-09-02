# from quantization.quant_bvae import BetaVae, Encoder
from bvae import BetaVae, Encoder
from torchsummary import summary
from constants import *
import torch
import copy 
# from torchviz import make_dot

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

if __name__ == '__main__':
    
    n_latent = N_LATENT
    beta = BETA
    input_dimensions = INPUT_DIMENSIONS
    batch = 2
    n_channels = N_CHANNELS
    grayscale = GRAYSCALE

    model=BetaVae(
    N_LATENT,
    BETA,
    N_CHANNELS,
    INPUT_DIMENSIONS,
    batch=2)

    encoder = Encoder(
    n_latent=N_LATENT,
    n_chan=N_CHANNELS,
    input_d=INPUT_DIMENSIONS)

    # COMPLETE MODEL 
    # weights = '/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/base_models/base_4_model_100_epochs/base_4_bvae_n30_b1.4_ch3_224x224.pt'
    
    # ENCODER ONLY MODEL
    weights = '/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/encoder_only/base_4_bvae_n30_b1.4_ch3_224x224.pt'

    # DYNAMIC QUANT MODEL
    # weights = '/Users/aditya/Desktop/FYP/project/code/fyp/weight_files/quantized_models/dynamic_quant.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.load_state_dict(copy.deepcopy(torch.load(weights, device)))

    # print(count_parameters(model))
    summary(encoder, (3, 224, 224))
    # dummy = torch.zeros([1, 3, 224, 224])
    # yhat = network(dummy)
    # make_dot(yhat, params=dict(list(network.named_parameters()))).render("net_vis", format="png")
    # print(network)

    # module = network.enc_conv1
    # print(list(module.named_parameters()))  
    # print(list(module.named_buffers()))