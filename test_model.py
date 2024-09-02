
from torch.nn import Module
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import torch
import copy
import time
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns
import matplotlib

sns.set_theme()
# from knowledge_distilation.student_arch_2 import Mod2BetaVae
# from knowledge_distilation.student_arch_1 import ModBetaVae

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from bvae import BetaVae
from utils.data_loader import get_data_loader
from constants import * 

def test_model(model: Module, test_loader: DataLoader, beta:float) -> None:
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        model.eval()

        batch_ce_losses = 0
        batch_kl_losses = 0
        batch_total_losses = 0 

        columns = ['test_ce_loss', 'test_kl_loss', 'test_total_loss']

        test_results = pd.DataFrame(columns=columns)
        times = []
        
        for i, data in enumerate(test_loader):
            batch_losses = dict.fromkeys(columns)
            input, _ = data
            num_samples = len(input)

            input = input.to(device)

            # start_time = time.time()
            out, mu, logvar = model(input)
            # end_time = time.time()
            # if (end_time - start_time < 0.175):
            #     times.append(end_time - start_time)
            
            plt.imshow(input[0].permute(1, 2, 0))
            plt.show()
            ## Display output Image 
            plt.imshow(out[0].permute(1, 2, 0))
            plt.show()

            kl_loss = torch.mul(
                input=torch.sum(mu.pow(2) + logvar.exp() - logvar - 1),
                other=0.5)

            ce_loss = torch.nn.functional.binary_cross_entropy(
                input=out,
                target=input,
                size_average=False)

            loss = ce_loss + torch.mul(kl_loss, beta)

            batch_ce_losses = ce_loss.item()
            batch_kl_losses = kl_loss.detach().numpy()
            batch_total_losses = loss.detach().numpy()

            batch_losses['test_ce_loss'] = batch_ce_losses/num_samples
            batch_losses['test_kl_loss'] = batch_kl_losses/num_samples
            batch_losses['test_total_loss'] = batch_total_losses/num_samples

            test_results = test_results.append(batch_losses, ignore_index=True) 
            print('Testing image: [{} / {}]'.format(i, len(test_loader)), end = "\r" )
            break
            if i == 10:
                print()
                break 
                

    # fig = plt.figure(figsize =(10, 7))
 
    # print(statistics.mean(times))
    # print(statistics.stdev(times))
    # # Creating plot
    # plt.boxplot(times)
    
    # # show plot
    # plt.show()
    # test_results.to_csv(dir, index=False)
    avg_ce_loss = test_results['test_ce_loss'].mean()
    avg_kl_loss = test_results['test_kl_loss'].mean()
    avg_total_loss = test_results['test_total_loss'].mean()

    print('Avg_ce_loss: {}'.format(avg_ce_loss))
    print('Avg_kl_loss: {}'.format(avg_kl_loss))
    print('Avg_total_loss: {}'.format(avg_total_loss))
    
    return avg_ce_loss, avg_kl_loss, avg_total_loss


#####################################################################
# Testing Model
#####################################################################

if __name__ == '__main__':

    data_file_prefix = "Train0_4"
    test_loader, _, _ = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH + data_file_prefix, BATCH, split=0.8)
  
    print("----------------------------------------")
    print("Test Set: {}".format(len(test_loader)*BATCH))
    print("----------------------------------------")

    model=BetaVae(
        N_LATENT,
        BETA,
        N_CHANNELS,
        INPUT_DIMENSIONS,
        BATCH)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = base_model_weights
   
    model.load_state_dict(copy.deepcopy(torch.load(weights, device)))
    # avg_ce_loss, avg_kl_loss, total_loss = test_model(model, test_loader, BETA)
    # test_model(model, test_loader, BETA)
    my_dict = {}
    a = np.array([])
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'weight' in name and 'bn' not in name and 'mu' not in name and 'var' not in name:
            a = torch.flatten(param).detach().numpy()
            print(a)
            
            count = np.count_nonzero(a > 0.01)
            count2  = np.count_nonzero(a < -0.01)
            my_dict[name.split('.')[0]] = count+ count2
            # my_dict[name.split('.')[0]] = len(torch.flatten(param).detach().numpy())
    print(my_dict)
    keys = list(my_dict.keys())
    # get values in the same order as keys, and parse percentage values
    vals = [my_dict[k] for k in keys]
    g = sns.barplot(x=keys, y=vals)
    g.set_yscale("log")
    plt.xticks(rotation=45)
    matplotlib.pyplot.show()
        # print(name)#, torch.flatten(param).detach().numpy())
        # a = np.append(a, torch.flatten(param).detach().numpy())
        # sns.histplot(a, binrange = (-0.1,0.1))#np.histogram_bin_edges(a, bins='auto', range=(0, 1)))
        # matplotlib.pyplot.show()
        # a = np.array([])
        # if i ==3:
        #     break
    # plt.boxplot(a, showfliers=False)
    
    # show plot
    # plt.show()
    df_describe = pd.DataFrame(a)
    # print(df_describe.boxplot())#by='X')
    print(df_describe.describe())
    # matplotlib.boxplot(a, showfliers=False)
    # print(model.times)
    # total = 0
    # for layer, t in model.times.items():
    #     print('{}'.format(t))
    #     total += t

    # print(total)

    ####################################################################
    # Testing for Puning
    #####################################################################

    # parent_dir = '/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/layerwise_pruning/'
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # main_columns = ['percentage', 'test_ce_loss', 'test_kl_loss', 'test_total_loss']

    # for i in sorted(os.listdir(parent_dir)):
    #     main_results = pd.DataFrame(columns=main_columns)
    #     k =0
    #     for j in sorted(os.listdir(parent_dir+i)):

    #         model.load_state_dict(copy.deepcopy(torch.load(parent_dir+i+'/'+j, device)))
    #         # if not os.path.isdir('results/layerwise_pruning/{}'.format(i)):
    #         #     os.makedirs('results/layerwise_pruning/{}'.format(i))
    #         print("----------------------------------------")
    #         print("{}/{}".format(i, j))
    #         print("----------------------------------------")

    #         avg_ce_loss, avg_kl_loss, total_loss =test_model(model, test_loader, BETA, 'results/layerwise_pruning/{}/{}.csv'.format(i, j.split('.')[0]))
        
    #         row = dict.fromkeys(main_columns)
    #         perc = (j.split('_')[-1]).split('.')[0]
    #         if len(perc) == 2:
    #             perc = '0{}'.format(perc)
    #         row['percentage'] = perc
    #         row['test_ce_loss'] = avg_ce_loss
    #         row['test_kl_loss'] = avg_kl_loss
    #         row['test_total_loss'] = total_loss
    #         print(main_results.columns[0])
            
    #         main_results = main_results.append(row, ignore_index=True)

    #     main_results = main_results.sort_values(by='percentage')
    #     print(main_results)
    #     main_results.to_csv('results/layerwise_pruning/{}.csv'.format(i), index=False)
        
