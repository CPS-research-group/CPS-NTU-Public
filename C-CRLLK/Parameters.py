import torch
import os
import numpy as np

# debug
SYS_DEBUG_MODE = False
NN_DEBUG_MODE = False
SINGLE_DEBUG_MODE = True

# ==========
# Version
train_version = '11_00_continuous_lagrange_zigzag'
load_version = '11_00_continuous_lagrange_zigzag'
# (do not change)
project_name = 'Duckie_RL_00'
train_model_path = 'log/' + train_version + '/model'
load_model_path = 'log/' + load_version + '/model'
save_render_path = 'log/' + load_version + '/render'

# Training Config
BOOL_RCPO = True
BOOL_TRAINING = True
BOOL_RENDER = False
# (recover)
BOOL_LOAD_MODEL = False
# (save)
BOOL_SAVE_RENDER = False

# Training Structure
NUM_RLRUNNER = 4 # 4
NUM_WORKER = 1
MAX_BUFFER_PER_RUNNER = 1
MIN_BUFFER_LENGTH = 4
MAX_NUM_EPISODE = 1e8
SAVE_EPISODES = 1e4
EPSIODE_PER_KILL = 1e4

# RL Parameters
LEARNING_RATE = 2.e-5
BETAS = (0.9, 0.999)
GAMMA = 0.95
EGREEDY = 0
SEED_NUM = 0
# ppo-ac
EPS_CLIP = 0.2
K_EPOCH_PPO = 1
# Lagrange Multiplier
LAGRANGE_LR_LANE =  2.e-5
LAGRANGE_LR_COLL =  2.e-5
LAGRANGE_LR_CROS =  2.e-5
LAGRANGE_TARGET_LANE = 0.5
LAGRANGE_TARGET_COLL = 0.01
LAGRANGE_TARGET_CROS = 0.01
# Non-RCPO
LAGRANGE_FIXED_LANE = 1
LAGRANGE_FIXED_COLL = 0.4
LAGRANGE_FIXED_CROS = 1
REWARD_FACTOR = 1

# Env Parameters
STATE_SIZE = (30, 40)
CHANNAL = 3
ACTION_SIZE = 2
MAX_NUM_STEPS = 512  # 1024
# Reward (fit to env)
TERMINIAL_REWARD = 0  # -200
COLLIDE_FACTOR = 100 # 100
CROSS_FACTOR = 10 # 100
LANE_FACTOR = 10
SAFE_FACTOR = 0.9
MOVE_REWARD = 1
LANE_REWARD = 1

# Net Config
GRIDIANT_CLIP = 10.0
DROP_PROB = 0.0
NET_SIZE = 256
# Net Adjust (fit to env)
ACTOR_MEAN_FACTOR = 1.0
ACTOR_SIGMA_FACTOR = 1.0
CITIC_NET_FACTOR = 1.0
VARIANCE_BOUNDARY_MIN = 0.1
VARIANCE_BOUNDARY_MAX = 0.6
ENTROPY_FACTOR = 0.001  # 0.01

# ==========
# Town Config
DUCKIETOWN = ['Duckietown-straight_road-v0',
              'Duckietown-4way-v0',
              'Duckietown-udem1-v0',
              'Duckietown-small_loop-v0',
              'Duckietown-small_loop_cw-v0',
              'Duckietown-zigzag_dists-v0',
              'Duckietown-loop_obstacles-v0',
              'Duckietown-loop_pedestrians-v0',
              'Duckietown-concave_loop_5x5-v0',
              'Duckietown-concave_loop_7x4-v0',
              'Duckietown-zigzag_without_obj-v0']
TOWN = DUCKIETOWN[10] # 0 3 5 8 9 10

# Device
CPU_ONLY = True
# (do not change)
TORCH_CPU = torch.device("cpu")
RUNNER_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() and not CPU_ONLY else torch.device("cpu")
DRIVER_DEVICE = torch.device("cuda:0") if torch.cuda.is_available() and not CPU_ONLY else torch.device("cpu")
NUM_CPU = os.cpu_count()
NUM_GPU = int(torch.cuda.device_count()) if torch.cuda.is_available() and not CPU_ONLY else 0
CPU_PER_RUNNER = int(np.floor(NUM_CPU / NUM_RLRUNNER))
GPU_PER_RUNNER = 1.0 * NUM_GPU / (NUM_RLRUNNER + 1) if torch.cuda.is_available() and not CPU_ONLY else 0
