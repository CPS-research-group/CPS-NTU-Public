from bvae import BetaVae
from utils.data_loader import get_data_loader
from constants import * 

print(f'Starting training for input size {INPUT_DIMENSIONS}')
print(f'beta={BETA}')
print(f'n_latent={N_LATENT}')
print(f'batch={BATCH}')
print(f'Using data set {TRAIN_DATAPATH}')

train_loader, val_loader, _ = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, 0.8)

print("Train Set: {}".format(len(train_loader)))
print("Val Set: {}".format(len(val_loader)))

model=BetaVae(
    N_LATENT,
    BETA,
    N_CHANNELS,
    INPUT_DIMENSIONS,
    BATCH)      

model.train_n_validate(
    train_loader, 
    val_loader,
    epochs=10,
    weights_file='/Users/aditya/Desktop/FYP/project/code/fyp/models/base_model/'+f'new_bvae_n{N_LATENT}_b{BETA}_ch{N_CHANNELS}_'
                    f'{"x".join([str(i) for i in INPUT_DIMENSIONS])}.pt', 
    file_prefix='base_bvae'
    lr = 1e-5)


def model_train(model, train_loader, val_loader, epochs, weights_file):
    model.train_n_validate(
        train_loader, 
        val_loader,
        epochs=epochs,
        weights_file=weights_file)



