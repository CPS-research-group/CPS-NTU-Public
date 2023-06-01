import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.patches import Patch

N_ITER=120

fig, ax = plt.subplots(1,4)
for idx, dt in enumerate(['0.5', '0.333333', '0.25', '0.2']):
    data = np.load(f'bak{dt}.npy.npz')
    data_bayes = np.load(f'bak{dt}_bayes.npy.npz')
    naive_med = np.percentile(-data['naive_bayes'][:,0:N_ITER], 50, axis=0)
    naive_lower = np.percentile(-data['naive_bayes'][:,0:N_ITER], 25, axis=0)
    naive_upper = np.percentile(-data['naive_bayes'][:,0:N_ITER], 75, axis=0)
    print(data['bayes_opt_ucb'])
    bayes_opt_med = np.percentile(-data_bayes['bayes_opt_ucb'][:,0:N_ITER], 50, axis=0)
    bayes_opt_lower = np.percentile(-data_bayes['bayes_opt_ucb'][:,0:N_ITER], 25, axis=0)
    bayes_opt_upper = np.percentile(-data_bayes['bayes_opt_ucb'][:,0:N_ITER], 75, axis=0)

    grid_med = np.percentile(-data['grid_search'][:,0:N_ITER], 50, axis=0)
    grid_lower = np.percentile(-data['grid_search'][:,0:N_ITER], 25, axis=0)
    grid_upper = np.percentile(-data['grid_search'][:,0:N_ITER], 75, axis=0)

    ax[idx].plot(np.linspace(0, N_ITER, N_ITER), naive_med, color='C0', alpha=1)
    ax[idx].fill_between(np.linspace(0, N_ITER, N_ITER), naive_lower, naive_upper, color='C0', alpha=0.25)
    
    ax[idx].plot(np.linspace(0, N_ITER, N_ITER), bayes_opt_med, color='C1', alpha=1)
    ax[idx].fill_between(np.linspace(0, N_ITER, N_ITER), bayes_opt_lower, bayes_opt_upper, color='C1', alpha=0.25)

    ax[idx].plot(np.linspace(0, N_ITER, N_ITER), grid_med, color='C2', alpha=1)
    ax[idx].fill_between(np.linspace(0, N_ITER, N_ITER), grid_lower, grid_upper, color='C2', alpha=0.25)

    ax[idx].set_xlabel('Iteration')
    ax[idx].set_ylabel('Minimum Found Risk')
    #plt.ylim(0, 0.3)
    legend_elements = [
        Patch(facecolor='C2', label='Grid Search'),
        Patch(facecolor='C0', label='Vanilla Bayes'),
        Patch(facecolor='C1', label='Ours'),
    ]
    
plt.legend(handles=legend_elements)
plt.show()

