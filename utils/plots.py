import matplotlib.pyplot as plt
import pandas as pd

def plot_loss():
    losses = pd.read_csv('/Users/aditya/Desktop/FYP/project/code/ICRTS2022/experiments/base_models/base_4_model_100_epochs/base_4_out.csv')

    losses = pd.read_csv('/Users/aditya/Desktop/base_4_out.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plt.figure(figsize=(10,5))
    # plt.title("Training & Val Losses")

    ax1.plot(losses['train_ce_loss'],label="train_ce_loss")
    ax1.plot(losses['train_total_loss'],label="train_total_loss")

    ax2.plot(losses['val_ce_loss'],label="val_ce_loss")
    ax2.plot(losses['val_total_loss'],label="val_total_loss")


    # ax1.plot(losses['train_kl_loss'],label="train_kl_loss")
    # ax1.plot(losses['val_kl_loss'],label="val_kl_loss")
    

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Reconstruction Loss")
    ax1.legend()


    # ax2.plot(losses['val_ce_loss'],label="val_ce_loss")
    # # ax2.plot(losses['val_kl_loss'],label="val_kl_loss")
    # ax2.plot(losses['val_total_loss'],label="val_total_loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Reconstruction Loss")
    ax2.legend()

    plt.show()

def plot_prune_loss():

    total_losses = [81265, 81171, 81125, 81368 ,81939, 82558 ,81964 ,83375 ,86208 ,95523, 98243, 101846, 106030, 108571]
    pruning = list(['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.92', '0.94', '0.96', '0.98'])

    fig = plt.figure(figsize = (10, 5))
    plt.bar(pruning, total_losses, width = .4)
    plt.xlabel("Amount Pruned")
    plt.ylabel("Total Recon Loss")
    plt.title("Total Loss vs Amount Pruned")
    plt.show()

# plot_prune_loss()

plot_loss()


def plot_composition():
    return
