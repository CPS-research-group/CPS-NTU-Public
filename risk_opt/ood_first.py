from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os
import math

# TENSOR FORMAT:
# YOLO Thresh, YOLO Size
OOD_SIZES = list(range(8,232,8))
SPACE = [100, len(OOD_SIZES), 100, 14]
RISK = np.zeros(SPACE)


# OOD FUNCTIONAL
fp_bvae = pd.read_csv('../bvae/fp.csv').to_numpy()[:, 1:]
fn_bvae = pd.read_csv('../bvae/fn.csv').to_numpy()[:, 1:]
tp_bvae = pd.read_csv('../bvae/tp.csv').to_numpy()[:, 1:]
tn_bvae = pd.read_csv('../bvae/tn.csv').to_numpy()[:, 1:]

# TENSOR FORMAT:
# OOD Size, OOD Thresh, YOLO Size, YOLO Thresh
RISK = np.zeros([100, 15, 100, 14])

palpha_bvae = fp_bvae / (fp_bvae + tn_bvae)
pbeta_bvae = tp_bvae / (tp_bvae + fn_bvae)
pgamma_bvae = fn_bvae / (fn_bvae + tp_bvae)
pdelta_bvae = tn_bvae / (tn_bvae + fp_bvae)
PALPHA = np.zeros(SPACE)
PBETA = np.zeros(SPACE)
PGAMMA = np.zeros(SPACE)
PDELTA = np.zeros(SPACE)
for i in range(14):
    for j in range(100):
        PALPHA[:,:,j,i] = palpha_bvae
        PBETA[:,:,j,i] = pbeta_bvae
        PGAMMA[:, :, j, i] = pgamma_bvae
        PDELTA[:, :, j, i] = pdelta_bvae



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
for i in range(len(OOD_SIZES)):
    for j in range(100):
        PAID[j,i,:,:] = paid
        PAOOD[j,i,:,:] = paood
        PBID[j,i,:,:] = pbid
        PBOOD[j,i,:,:] = pbood
        PCID[j,i,:,:] = pcid
        PCOOD[j,i,:,:] = pcood
        PDID[j,i,:,:] = pdid
        PDOOD[j,i,:,:] = pdood





# TIMING DATA
#FPS=5
TDS = [0.5, 0.333333, 0.25, 0.2]
#fig, ax = plt.subplots(4,len(TDS))
for IDX, TD in enumerate(TDS):
    SAMPLES = 1000
    t_yolo = np.zeros((14, SAMPLES), dtype=np.float32)
    t_ood = np.zeros((len(OOD_SIZES), SAMPLES), dtype=np.float32)
    for jdx, yolo_size in enumerate(range(64,512,32)):
        yolo_df = pd.read_csv(os.path.expanduser(f'../../timing_results/yolo_only/yolo{yolo_size}_yoloONLY_{yolo_size}.csv'))
        yolo_times = (yolo_df['End'] - yolo_df['Start']).to_numpy()
        if yolo_times.size < SAMPLES:
            yolo_times = np.repeat(yolo_times, math.ceil(SAMPLES / yolo_times.size))
        t_yolo[jdx, :] = yolo_times[:SAMPLES]
    for idx, ood_size in enumerate(OOD_SIZES):
        ood_df = pd.read_csv(os.path.expanduser(f'../../timing_results/bvae/ood{ood_size}_ood{ood_size}bvae.csv'))
        ood_times = (ood_df['End'] - ood_df['Start']).to_numpy()
        if ood_times.size < SAMPLES:
            ood_times = np.repeat(ood_times, math.ceil(SAMPLES / ood_times.size))
        t_ood[idx, :] = ood_times[:SAMPLES]
    total_t = np.zeros((len(OOD_SIZES), 14, SAMPLES))
    for i in range(len(OOD_SIZES)):
        for j in range(14):
            total_t[i, j, :] = t_ood[i, :] + t_yolo[j, :]
    p_miss_yolo = np.zeros((len(OOD_SIZES), 14))
    p_miss_ood = np.zeros((len(OOD_SIZES), 14))
    for i in range(len(OOD_SIZES)):
        for j in range(14):
            p_miss_ood[i, j] = np.count_nonzero(t_ood[i, :] > TD) / SAMPLES
            p_miss_yolo[i, j] = np.count_nonzero(total_t[i, j, :] > TD) / SAMPLES
    PE = np.zeros(SPACE)
    PEPSILON = np.zeros(SPACE)
    for i in range(100):
        for j in range(100):
            PE[i, :, j, :] = p_miss_yolo
            PEPSILON[i, :, j, :] = p_miss_ood

    POOD = 1e-6 * np.ones(SPACE)



    #############################
    # RISK WITH NO OOD DETECTOR #
    #############################
    #E1 = (1 - POOD) * PCID * PDELTA + (1 - POOD) * PCID * PEPSILON + (1 - POOD) * PBID * PDELTA * PEPSILON + (1 - POOD) * PBID * PE * PEPSILON - (1 - POOD) * PCID * PDELTA * PEPSILON - (1 - POOD) * PBID * PDELTA * PE * PEPSILON + POOD * PCOOD * PEPSILON + POOD * PCOOD * PGAMMA + POOD * PBOOD * PE * PEPSILON + POOD * PBOOD * PE * PGAMMA - POOD * PCOOD * PEPSILON * PGAMMA - POOD * PBOOD * PGAMMA * PE * PEPSILON
    #E2 = (1 - POOD) * PAID * (1 - PE) + (1 - POOD) * PAID * PALPHA * (1 - PEPSILON) + (1 - POOD) * PDID * PALPHA * (1 - PEPSILON) + (1 - POOD) * PAID * PALPHA * (1 - PE) * (1 - PEPSILON) + POOD * PAOOD * (1 - PE) + POOD * PAOOD * PBETA * (1 - PEPSILON) + POOD * PDOOD * PBETA * (1 - PEPSILON) - POOD * PAOOD * PBETA * (1 - PE) * (1 - PEPSILON)
    #RISK = E1 + E2
    PXC = (POOD * PCOOD + (1-POOD) * PCID) * (1 - PE)
    PXDELTA = (1 -POOD) * PDELTA * (1 - PEPSILON)
    PXEPSILON = PEPSILON
    PXGAMMA = POOD * PGAMMA * (1 - PEPSILON)
    A = PXDELTA + PXGAMMA + PXEPSILON
    B = 0
    E1 = PXC * (A-B) + PE * B
    PXA = (POOD * PAOOD + (1-POOD) * PAID) * (1-PE)
    PXALPHA = (1-POOD) * PALPHA * (1-PEPSILON)
    PXBETA = POOD * PBETA * (1 - PEPSILON)
    E2 = PXA + PXALPHA + PXBETA - PXA * (PXALPHA + PXBETA - 0 - 0)
    RISK = 0.5 * (3 * E1 + E2)


    U = np.zeros(SPACE)
    for i in range(len(OOD_SIZES)):
        for j in range(14):
            U[:,i,:,j] = np.average(t_yolo[j,:] + t_ood[i,:])/TD
    f = {0.5: 0.6782, 0.333333:0.6838, 0.25:0.9252, 0.2:1.0}
    #RISK[U >= f[TD]] = 1

    #np.save(f'risk_ood_first_{TD}.npy', RISK)
    #np.save(f'U_ood_first_{TD}.npy', U)


    print(f'###### COMBINED RISK {TD} ######')
    print(f'MINIMUM RISK: {np.amin(RISK)}')
    print(f'MAXIMUM RISK: {np.amax(RISK)}')
    min_risk = np.unravel_index(np.argmin(RISK, axis=None), RISK.shape)
    print(f'OOD THRESH: {min_risk[0]}')
    print(f'OOD SIZE: {8 + 8 * min_risk[1]}')
    print(f'YOLO THRSH: {min_risk[2]}')
    print(f'YOLO SIZE: {64 + 32 * min_risk[3]}')
    print(f'E1@MIN: {E1[min_risk]}')
    print(f'E2@MIN: {E2[min_risk]}')
    print(f'U AVG @ MIN RISK: {np.average(t_yolo[min_risk[3],:] + t_ood[min_risk[1], :])/TD}')
    print(f'MIN E1: {np.amin(E1)}')
    print(f'MAX E1: {np.amax(E1)}')
    min_e1 = np.unravel_index(np.argmin(E1, axis=None), E1.shape)
    print(f'MIN E2: {np.amin(E2)}')
    print(f'MAX E2: {np.amax(E2)}')
    min_e2 = np.unravel_index(np.argmin(E2, axis=None), E2.shape)

    Y = np.linspace(0, 1, 100)
    X = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(X, Y)

    Y0 = np.array(range(8,232,8))
    X0 = np.array(range(64,512,32))
    X0, Y0 = np.meshgrid(X0, Y0)

    X2 = np.array(range(8,232,8))
    Y2 = np.array(range(64,512,32))
    X2, Y2 = np.meshgrid(X2, Y2)
    #fig, ax = plt.subplots(1, 3)
    fig, ax = plt.subplots(2,2)
    levels = ax[0][0].contourf(X0, Y0, RISK[min_risk[0],:,min_risk[2],:], extent=[8,224,32,480], vmin=0, vmax=0.4)
    #fig.colorbar(levels)
    ax[0][0].set_ylabel('OOD input size')
    ax[0][0].set_xlabel('YOLO input size')
    ax[0][0].set_title(f'Risk')

    ax[1][0].contourf(X2, Y2, np.transpose(PE[min_risk[0], :, min_risk[2], :]), extent=[8,224,32,480], vmin=0, vmax=1)
    ax[1][0].set_ylabel('OOD input size')
    ax[1][0].set_xlabel('YOLO input size')
    ax[1][0].set_title(f'$P(E_e)$')

    ax[0][1].contourf(X0, Y0, E1[min_risk[0], :, min_risk[2], :], extent=[8,224,32,480], vmin=0, vmax=1e-11)
    ax[0][1].set_ylabel('OOD input size')
    ax[0][1].set_xlabel('YOLO input size')
    ax[0][1].set_title('$P(E_0)$')


    ax[1][1].contourf(X0, Y0, E2[min_risk[0], :, min_risk[2], :], extent=[8,224,32,480], vmin=0, vmax=1)
    ax[1][1].set_ylabel('OOD input size')
    ax[1][1].set_xlabel('YOLO input size')
    ax[1][1].set_title(f'$P(E_1)$')

    plt.show()


#plt.show()


