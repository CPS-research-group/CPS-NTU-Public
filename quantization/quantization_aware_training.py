from calendar import EPOCH
import torch
import pandas as pd
from utils.data_loader import get_data_loader
from constants import *

# QAT follows the same steps as static quantization,
# with the exception of the training loop before you actually convert the model to its quantized version

def train_model(model, 
                train_loader,
                epochs,
                weights_file):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print (f'Using device: {device}')
        network = model.to(device)
        network.eval()

        optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

        num_samples = len(train_loader)
        print(f'Num Samples: {num_samples}')

        columns = ['train_ce_loss', 'train_kl_loss', 'train_total_loss',
                   'val_ce_loss', 'val_kl_loss', 'val_total_loss']

        train_results = pd.DataFrame(columns=columns)
        avg_losses = dict.fromkeys(columns)
        
        for epoch in range(epochs):

            epoch_ce_loss = 0
            epoch_kl_loss = 0
            epoch_total_loss = 0
            print("Epoch: [{}/{}]".format(epoch+1,epochs))
            for i, data in enumerate(train_loader):
            
                print("Training Images: [{}/{}]".format(i+1,len(train_loader)), end='\r')
                input, _ = data
                input = input.to(device)
                out, mu, logvar = network(input)

                if epoch == 75:
                    for group in optimizer.param_groups:
                        group['lr'] = 1e-6

                kl_loss = torch.mul(
                    input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                    other=0.5)
                ce_loss = torch.nn.functional.binary_cross_entropy(
                    input=out,
                    target=input,
                    size_average=False)
                loss = ce_loss + torch.mul(kl_loss, model.beta)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_ce_loss += ce_loss.item()
                epoch_kl_loss += kl_loss.detach().numpy()
                epoch_total_loss += loss.detach().numpy()

            avg_losses['train_ce_loss'] = epoch_ce_loss/num_samples
            avg_losses['train_kl_loss'] = epoch_kl_loss/num_samples
            avg_losses['train_total_loss'] = epoch_total_loss/num_samples

            train_results = train_results.append(avg_losses, ignore_index=True)

            print(f"Epoch: {epoch}; Train_loss: {avg_losses['train_total_loss']}")
        print('Training finished, saving results csv & saving weights...')

        train_results.to_csv('qat_train_results/out.csv', index=False)
        torch.save(network.state_dict(), weights_file)


def qat_quantise(model, train=False):

    backend = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = 'qnnpack'    
    
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)

    if train:
        train_loader, _, _ = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, split=0.8)
        train_model(model, train_loader, epochs=100, weights_file='quantized_models/trained_qat.pt')

    model.eval()
    q_model = torch.quantization.convert(model, inplace=False)
    
    return q_model
