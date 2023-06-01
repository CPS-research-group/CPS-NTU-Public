from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

# TENSOR FORMAT:
# YOLO Thresh, YOLO Size
SPACE = [100, 14]
RISK = np.zeros(SPACE)

# YOLO FUNCTIONAL
ID_RANGE = range(2)
OOD_RANGE = range(2,10)
fp_count_id = 0
empty_id = 0 
fn_count_id = 0
have_id = 0
fp_count_ood = 0
empty_ood =0
fn_count_ood =0
have_ood = 0
for i in ID_RANGE:
    fp_count_id += pd.read_csv(f'yolo_res/fp_count_{i}.csv').to_numpy()[:, 1:]
    empty_id += pd.read_csv(f'yolo_res/empty_count_{i}.csv').to_numpy()[:, 1:]
    fn_count_id += pd.read_csv(f'yolo_res/fn_count_{i}.csv').to_numpy()[:, 1:]
    have_id += pd.read_csv(f'yolo_res/have_count_{i}.csv').to_numpy()[:, 1:]
for j in OOD_RANGE:
    fp_count_ood += pd.read_csv(f'yolo_res/fp_count_{j}.csv').to_numpy()[:, 1:]
    empty_ood += pd.read_csv(f'yolo_res/empty_count_{j}.csv').to_numpy()[:, 1:]
    fn_count_ood += pd.read_csv(f'yolo_res/fn_count_{j}.csv').to_numpy()[:, 1:]
    have_ood += pd.read_csv(f'yolo_res/have_count_{j}.csv').to_numpy()[:, 1:]
paid = fp_count_id / empty_id
paood = fp_count_ood / empty_ood
pbid = 1 - fn_count_id / have_id
pbood = 1 - fn_count_ood / have_ood
pcid = fn_count_id / have_id
pcood = fn_count_ood / have_ood
pdid = 1 - paid
pdood = 1 - paood
PAID = np.zeros(SPACE)
PAOOD = np.zeros(SPACE)
PBID = np.zeros(SPACE)
PBOOD = np.zeros(SPACE)
PCID = np.zeros(SPACE)
PCOOD = np.zeros(SPACE)
PDID = np.zeros(SPACE)
PDOOD = np.zeros(SPACE)
PAID = paid
PAOOD = paood
PBID = pbid
PBOOD = pbood
PCID = pcid
PCOOD = pcood
PDID = pdid
PDOOD = pdood

# TIMING DATA
#FPS=5
TDS = [0.5, 0.333333,0.25, 0.2]
fig, ax = plt.subplots(3, len(TDS))
for IDX, TD in enumerate(TDS):
    SAMPLES = 48
    t_yolo = np.zeros((14, SAMPLES), dtype=np.float32)
    for jdx, yolo_size in enumerate(range(64,512,32)):
        yolo_df = pd.read_csv(os.path.expanduser(f'../../timing_results/yolo_only/yolo{yolo_size}_yoloONLY_{yolo_size}.csv'))
        #yolo_df = pd.read_csv(os.path.expanduser(f'../../timing_results/old/yolo{yolo_size}_{yolo_size}y128o.csv'))

        yolo_times = yolo_df['End'] - yolo_df['Start']
        t_yolo[jdx, :] = yolo_times.to_numpy()[:SAMPLES]
    p_miss_yolo = np.zeros(14)
    for j in range(14):
        p_miss_yolo[j] = np.count_nonzero(t_yolo[j, :] > TD) / SAMPLES
    PE = np.zeros(SPACE)
    for j in range(100):
        PE[j, :] = p_miss_yolo

    POOD = 1e-6 * np.ones(SPACE)


    #############################
    # RISK WITH NO OOD DETECTOR #
    #############################
    PEC = (POOD * PCOOD + (1- POOD) * PCID) * (1 - PE)
    PEE = PE
    PEA = (POOD * PAID + (1- POOD) * PAOOD) * (1 - PE)
    E1 = PEC + PEE
    E2 = PEA
    RISK = 3 * E1 + E2
    np.save(f'risk_yolo_only_{TD}.npy', RISK)
    np.save(f'e1_yolo_only_{TD}.npy', E1)
    np.save(f'e2_yolo_only_{TD}.npy', E2)


    print(f'###### YOLO ONLY ######')
    print(f'MINIMUM RISK: {np.amin(RISK)}')
    print(f'MAXIMUM RISK: {np.amax(RISK)}')
    min_risk = np.unravel_index(np.argmin(RISK, axis=None), RISK.shape)
    print(f'YOLO THRESH: {min_risk[0]}')
    print(f'YOLO SIZE: {64 + 32 * min_risk[1]}')
    print(f'E1@MIN: {E1[min_risk]}')
    print(f'E2@MIN: {E2[min_risk]}')
    print(f'U AVG @ MIN RISK: {np.average(t_yolo[min_risk[1],:])/TD}')
    print(f'MIN E1: {np.amin(E1)}')
    print(f'MAX E1: {np.amax(E1)}')
    min_e1 = np.unravel_index(np.argmin(E1, axis=None), E1.shape)
    print(f'YOLO THRESH: {64 + 32 * min_e1[1]}')
    print(f'YOLO SIZE: {64 + 32 * min_e1[1]}')
    print(f'MIN E2: {np.amin(E2)}')
    print(f'MAX E2: {np.amax(E2)}')
    min_e2 = np.unravel_index(np.argmin(E2, axis=None), E2.shape)
    print(f'YOLO THRESH: {64 + 32 * min_e2[1]}')
    print(f'YOLO SIZE: {64 + 32 * min_e2[1]}')

    Y = np.linspace(0, 1, 100)
    X = np.array(range(64,512,32))
    X, Y = np.meshgrid(X, Y)
    #fig, ax = plt.subplots(1, 3)
    #levels = ax[0][IDX].contourf(X, Y, E1, extent=[64,480,0,1], vmin=0, vmax=1)
    #fig.colorbar(levels)
    #ax[0][IDX].set_xlabel('YOLO input size')
    #ax[0][IDX].set_ylabel('YOLO th###### YOLO ONLY ######reshold')
    #ax[2][IDX].set_title(f'Risk')

    fig, ax = plt.subplots(1, 2)
    levels = ax[0].contourf(X, Y, E1, extent=[64,480,0,1], vmin=0, vmax=1)
    #fig.colorbar(levels)
    ax[0].set_xlabel('YOLO input size')
    ax[0].set_ylabel('YOLO threshold')
    ax[0].set_title('$P(E_0)$')
    
    ax[1].contourf(X, Y, E2, extent=[64, 480, 0, 1], vmin=0, vmax=1)
    ax[1].set_xlabel('YOLO input size')
    ax[1].set_ylabel('YOLO threshold')
    ax[1].set_title('$P(E_1)$')

    plt.show()   

#plt.show()


