from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

fp = pd.read_csv('../bvae/fp.csv').to_numpy()[:, 1:16]
fn = pd.read_csv('../bvae/fn.csv').to_numpy()[:, 1:16]
tp = pd.read_csv('../bvae/tp.csv').to_numpy()[:, 1:16]
tn = pd.read_csv('../bvae/tn.csv').to_numpy()[:, 1:16]

# TENSOR FORMAT:
# OOD Size, OOD Thresh, YOLO Size, YOLO Thresh
RISK = np.zeros([100, 15, 100, 14])

palpha = fp / (fp + tn)
pbeta = tp / (tp + fn)
pgamma = fn / (fn + tp)
pdelta = tn / (tn + fp)
PALPHA = np.zeros([100, 15, 100, 14])
PBETA = np.zeros([100, 15, 100, 14])
PGAMMA = np.zeros([100, 15, 100, 14])
PDELTA = np.zeros([100, 15, 100, 14])
for i in range(14):
    for j in range(100):
        PALPHA[:,:,j,i] = palpha
        PBETA[:,:,j,i] = pbeta
        PGAMMA[:, :, j, i] = pgamma
        PDELTA[:, :, j, i] = pdelta


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
#fp_count_id = pd.read_csv('fp_count_id.csv').to_numpy()[:, 1:]
#empty_id = pd.read_csv('empty_count_id.csv').to_numpy()[:, 1:]
#fn_count_id = pd.read_csv('fn_count_id.csv').to_numpy()[:, 1:]
#have_id = pd.read_csv('have_count_id.csv').to_numpy()[:, 1:]
#fp_count_ood = pd.read_csv('fp_count_ood.csv').to_numpy()[:, 1:]
#empty_ood = pd.read_csv('empty_count_ood.csv').to_numpy()[:, 1:]
#fn_count_ood = pd.read_csv('fn_count_ood.csv').to_numpy()[:, 1:]
#have_ood = pd.read_csv('have_count_ood.csv').to_numpy()[:, 1:]



# TENSOR SIZE
SPACE = [100, 15, 100, 14]
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
for i in range(15):
    for j in range(100):
        PAID[j, i, :, :] = paid
        PAOOD[j, i, :, :] = paood
        PBID[j, i, :, :] = pbid
        PBOOD[j, i, :, :] = pbood
        PCID[j, i, :, :] = pcid
        PCOOD[j, i, :, :] = pcood
        PDID[j, i, :, :] = pdid
        PDOOD[j, i, :, :] = pdood


SAMPLES = 48
t_ood = np.zeros((15, 14, SAMPLES), dtype=np.float32)
t_yolo = np.zeros((15,14, SAMPLES), dtype=np.float32)
for idx, ood_size in enumerate(range(32,512,32)):
    for jdx, yolo_size in enumerate(range(64,512,32)):
        try:
            ood_df = pd.read_csv(os.path.expanduser(f'../../timing_results/old/ood{ood_size}_{yolo_size}y{ood_size}o.csv'))
            yolo_df = pd.read_csv(os.path.expanduser(f'../../timing_results/old/yolo{yolo_size}_{yolo_size}y{ood_size}o.csv'))
            ood_times = ood_df['End'] - ood_df['Start']
            t_ood[idx, jdx, :] = ood_times.to_numpy()[:SAMPLES]
            yolo_times = yolo_df['End'] - yolo_df['Start']
            t_yolo[idx, jdx, :] = yolo_times.to_numpy()[:SAMPLES]
        except FileNotFoundError:
            t_ood[idx, jdx, :] = np.ones(SAMPLES)
            t_yolo[idx, jdx, :] = np.ones(SAMPLES)
total_t = np.zeros((15, 14, SAMPLES))
for i in range(15):
    for j in range(14):
        total_t[i, j, :] = t_ood[i, j, :] + t_yolo[i, j, :]
FPS = 5
TD = 1 / FPS
# OOD FIRST
p_miss_yolo = np.zeros((15, 14))
p_miss_ood = np.zeros((15, 14))
for i in range(15):
    for j in range(14):
        p_miss_yolo[i, j] = np.count_nonzero(total_t[i, j, :] > TD) / SAMPLES
        p_miss_ood[i, j] = np.count_nonzero(t_ood[i, j, :] > TD) / SAMPLES

PE = np.zeros(SPACE)
PEPSILON = np.zeros(SPACE)
for i in range(100):
    for j in range(100):
        PE[i, :, j, :] = p_miss_yolo
        PEPSILON[i, :, j, :] = p_miss_ood

POOD = 1e-9 * np.ones(SPACE)
POOD = np.zeros(SPACE)
#E1 = -PBID * PDELTA * PE * PEPSILON * (POOD - 1) + PBOOD * PE * PEPSILON * POOD + PBOOD * PEPSILON * PGAMMA * POOD - PCID * PDELTA * (POOD - 1) + PCOOD * PGAMMA * POOD + PEPSILON * (PCID + PCOOD)
#E1 = (PCID * PDELTA) + (PCID * PEPSILON) + (PCOOD * PEPSILON) + (PCOOD * PGAMMA) + (PDELTA * PE) + (PE * PEPSILON) + (PE + PGAMMA)
#E1 = PCID * PDELTA + PCID * (PEPSILON - PDELTA * PEPSILON) + PCOOD * (PEPSILON - PGAMMA * PEPSILON) + PCOOD * PGAMMA + PBID * PDELTA * PEPSILON + PBID * PE * PEPSILON + PBOOD * PE * PEPSILON + PBOOD * PE * PGAMMA
#(c_id & delta) | (c_id & epsilon) | (c_ood & epsilon) | (c_ood & gamma) | (delta & e) | (e & epsilon) | (e & gamma)
E1 = (1 - POOD) * PCID * PDELTA + (1 - POOD) * PCID * PEPSILON + (1 - POOD) * PBID * PDELTA * PEPSILON + (1 - POOD) * PBID * PE * PEPSILON - (1 - POOD) * PCID * PDELTA * PEPSILON - (1 - POOD) * PBID * PDELTA * PE * PEPSILON + POOD * PCOOD * PEPSILON + POOD * PCOOD * PGAMMA + POOD * PBOOD * PE * PEPSILON + POOD * PBOOD * PE * PGAMMA - POOD * PCOOD * PEPSILON * PGAMMA - POOD * PBOOD * PGAMMA * PE * PEPSILON
#E2 = -(PALPHA + POOD) * (PCID + PCOOD + PDID + PDOOD) * (PEPSILON - 1) * (PBETA - POOD + 1)
E2 = (1 - POOD) * PAID * (1 - PE) + (1 - POOD) * PAID * PALPHA * (1 - PEPSILON) + (1 - POOD) * PDID * PALPHA * (1 - PEPSILON) + (1 - POOD) * PAID * PALPHA * (1 - PE) * (1 - PEPSILON) + POOD * PAOOD * (1 - PE) + POOD * PAOOD * PBETA * (1 - PEPSILON) + POOD * PDOOD * PBETA * (1 - PEPSILON) - POOD * PAOOD * PBETA * (1 - PE) * (1 - PEPSILON)

RISK = E1 + E2

np.save('risk.npy', RISK)


#############################
# RISK WITH NO OOD DETECTOR #
#############################
E1_NO = ((1 - POOD) * PCID + POOD * PCOOD) + (((1 - POOD) * PBID + POOD * PBOOD) * PE)
E2_NO = (1 - PE) * ((1 - POOD) * PAID + POOD * PAOOD)
RISK_NO = E1_NO + E2_NO
no_ood_risk = np.amin(RISK_NO)

print(f'###### CODESIGN #######')
print(f'MINIMUM RISK: {np.amin(RISK)}')
print(f'MAXIMUM RISK: {np.amax(RISK)}')
min_risk = np.unravel_index(np.argmin(RISK, axis=None), RISK.shape)
print(f'OOD THRESH: {min_risk[0]}')
print(f'OOD SIZE: {32 + 32 * min_risk[1]}')
print(f'YOLO THRESH: {min_risk[2]}')
print(f'YOLO SIZE: {64 + 32 * min_risk[3]}')

print(f'###### YOLO ONLY ######')
print(f'MINIMUM RISK: {np.amin(RISK_NO)}')
print(f'MAXIMUM RISK: {np.amax(RISK_NO)}')
min_risk_no = np.unravel_index(np.argmin(RISK_NO, axis=None), RISK.shape)
print(f'YOLO THRESH: {min_risk_no[2]}')
print(f'YOLO SIZE: {64 + 32 * min_risk_no[3]}')

fig, ax = plt.subplots(2, 2)
ax[0][0].plot(np.linspace(0,1,100), RISK[:, min_risk[1], min_risk[2], min_risk[3]])
ax[0][0].plot(np.linspace(0,1,100), no_ood_risk * np.ones(100))
ax[0][0].set_xlabel('OOD detection threshold')
ax[0][0].set_ylabel('Risk')
ax[0][0].set_title(f'OOD SIZE: {32 + 32 * min_risk[1]}; YOLO THRESH: {min_risk[2]}; YOLO SIZE: {64 + 32 * min_risk[3]}')

ax[0][1].plot(np.array(list(range(32, 512, 32))), RISK[min_risk[0], :, min_risk[2], min_risk[3]])
ax[0][1].plot(np.array(list(range(34, 512, 32))), no_ood_risk * np.ones(15))
ax[0][1].set_xlabel('OOD detector size')
ax[0][1].set_ylabel('Risk')
ax[0][1].set_title(f'OOD THRESH: {min_risk[0]}; YOLO THRESH: {min_risk[2]}; YOLO SIZE: {64 + 32 * min_risk[3]}')

ax[1][0].plot(np.linspace(0,1,100), RISK[min_risk[0], min_risk[1], :, min_risk[3]])
ax[1][0].plot(np.linspace(0,1,100), no_ood_risk * np.ones(100))
ax[1][0].set_xlabel('YOLO threshold')
ax[1][0].set_ylabel('Risk')
ax[1][0].set_title(f'OOD THRESH: {min_risk[0]}; OOD SIZE: {32 + 32 * min_risk[1]}; YOLO SIZE: {64 + 32 * min_risk[3]}')

ax[1][1].plot(np.array(list(range(64, 512, 32))), RISK[min_risk[0], min_risk[1], min_risk[2], :])
ax[1][1].plot(np.array(list(range(64, 512, 32))), no_ood_risk * np.ones(14))
ax[1][1].set_xlabel('YOLO size')
ax[1][1].set_ylabel('Risk')
ax[1][1].set_title(f'OOD THRESH: {min_risk[0]}; OOD SIZE: {32 + 32 * min_risk[1]}; YOLO THRESH: {min_risk[2]}')

plt.show()


