import os
import ray
import time
import torch
import wandb

import numpy as np
from PPO import PPO
from Runner import RLRunner
from Lagrange import LagrangeMethod
from Parameters import *

torch.set_num_threads(1)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ['VECLIB_MAXIMUM_THREADS'] = "1"
os.environ["PYGLET_DEBUG_GL"] = "True"

torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)

print("{0}  Hello World:)".format(time.asctime(time.localtime(time.time()))))
print("cpu device num:", NUM_CPU)
print("cuda device available:", torch.cuda.is_available())
print("cuda device num:", NUM_GPU)
ray.init(num_gpus=NUM_GPU)

# Create directories
if not os.path.exists(train_model_path):
    os.makedirs(train_model_path)


def write_to_board(update_result, recorder, curr_episode, rcpo):
    start_loss, end_loss, weight_variance, grad_norms = update_result
    # Reward Logger
    for batch_index in range(len(rcpo.driver_memory.batch_buffer)):
        total_return = sum(rcpo.driver_memory.batch_buffer[batch_index]['return'])
        avg_return = total_return/len(rcpo.driver_memory.batch_buffer[batch_index]['return'])
        # _episode = batch_index + curr_episode - NUM_RLRUNNER
        wandb.log({
            'Return/Total_Return': total_return,
            'Return/Avg_Return': avg_return,
        },
        step=batch_index + curr_episode - NUM_RLRUNNER)

    if int(recorder['bool_learn']) != 0:
        wandb.log({
            # Loss
            # start_loss
            'Loss/total_loss': start_loss['total_loss'],
            'Loss/actor_loss': start_loss['actor_loss'],
            'Loss/critic_loss': start_loss['critic_loss'],
            'Loss/entropy_loss': start_loss['entropy_loss'],
            # end_loss
            # 'Loss/value_loss/end_loss': end_loss['total_loss'],
            # 'Loss/policy_loss/end_loss': end_loss['actor_loss'],
            # 'Loss/valid_loss/end_loss': end_loss['critic_loss'],
            # 'Loss/block_loss/end_loss': end_loss['entropy_loss'],
            # weight_variance
            'Loss/weight_variance': weight_variance,
            'Loss/grad_norms': grad_norms,
    
            # Recorder
            'Env/total_step': recorder['total_step']/NUM_RLRUNNER,
            'Env/total_reward': recorder['total_reward']/NUM_RLRUNNER,
            'Env/total_done': recorder['total_done']/NUM_RLRUNNER,
            # 'Env/bool_learn': recorder['bool_learn']/NUM_RLRUNNER,
            'Env/total_distance': recorder['total_distance']/NUM_RLRUNNER,
            'Env/total_angle': recorder['total_angle']/NUM_RLRUNNER,
            'Env/robot_speed': recorder['robot_speed']/NUM_RLRUNNER,
            'Env/total_lane': recorder['total_lane']/NUM_RLRUNNER,
            'Env/step_reward': recorder['total_reward']/recorder['total_step'],
    
            # Lagrange
            'Lagrange/multiplier_lane': rcpo.lagrange_multiplier_lane,
            'Lagrange/multiplier_coll': rcpo.lagrange_multiplier_coll,
            'Lagrange/multiplier_cros': rcpo.lagrange_multiplier_cros,
            'Lagrange/J_cost_lane': rcpo.J_cost_lane,
            'Lagrange/J_cost_coll': rcpo.J_cost_coll,
            'Lagrange/J_cost_cros': rcpo.J_cost_cros,
            'Lagrange/step_reward_hat': recorder['total_reward_hat']/recorder['total_step'],
            },
            step=curr_episode)

    


def main():
    # Login
    wandb.init(project=project_name, entity="sherwin-gao", name=train_version)
    print('(Driver)====== version:' + train_version + ' ======')
    print('(Driver)Ready to train! version: ', train_version)

    # Lagrange Parameters for Reward Constrain
    rcpo = LagrangeMethod(LAGRANGE_TARGET_LANE, LAGRANGE_TARGET_COLL, LAGRANGE_LR_CROS)

    # Driver network
    ppo = PPO(ACTION_SIZE, DRIVER_DEVICE)

    # Load the model to train:
    curr_episode = 0
    if BOOL_LOAD_MODEL:
        checkpoint = torch.load(load_model_path + '/checkpoint.pkl')
        ppo.global_network.load_state_dict(checkpoint['model_state_dict'])
        ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        curr_episode = checkpoint['epoch']
        rcpo.lagrange_multiplier_lane = checkpoint['lagrange_multiplier_lane']
        rcpo.lagrange_multiplier_coll = checkpoint['lagrange_multiplier_coll']
        rcpo.lagrange_multiplier_cros = checkpoint['lagrange_multiplier_cros']
        print('(Driver)Model Load. version:', load_version)
        print('(Driver)Train from episode:', curr_episode)
        print('(Driver)Train with lagrange_multiplier_lane:', rcpo.lagrange_multiplier_lane)
        print('(Driver)Train with lagrange_multiplier_coll:', rcpo.lagrange_multiplier_coll)
        print('(Driver)Train with lagrange_multiplier_cros:', rcpo.lagrange_multiplier_cros)
    else:
        print('(Driver)Model not Load. version: ', None)
        print('(Driver)Train from episode:', curr_episode)

    # Ray
    meta_agents = [RLRunner.remote(i, rcpo) for i in range(NUM_RLRUNNER)]
    # meta_agents = rl_agents
    weights = ppo.global_network.state_dict()
    # launch the first job
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(curr_episode, weights, rcpo))
        curr_episode += 1

    # Recorder
    best_reward = None
    # bug package: GLException(msg)
    restart_counter = 0

    # Run
    try:
        while True:
            # wait for job done
            # print(time.asctime(time.localtime(time.time())))
            # print('(Driver)waiting...')
            while len(job_list):
                done_id, job_list = ray.wait(job_list)
                info, runner_memory = ray.get(done_id)[0]

                # data collection
                ppo.driver_memory.push_memory(runner_memory)
                rcpo.driver_memory.push_memory(runner_memory)

            ppo.driver_memory.buffer_sort()
            update_result = ppo.driver_update()
            rcpo.lagrage_update()

            # log
            write_to_board(update_result, ppo.driver_memory.recorder, curr_episode, rcpo)

            # if save or not
            if curr_episode % SAVE_EPISODES < NUM_RLRUNNER:
                checkpoint = {'model_state_dict': weights,
                              'optimizer_state_dict': ppo.optimizer.state_dict(),
                              'epoch': curr_episode,
                              'lagrange_multiplier_lane':rcpo.lagrange_multiplier_lane,
                              'lagrange_multiplier_coll':rcpo.lagrange_multiplier_coll,
                              'lagrange_multiplier_cros':rcpo.lagrange_multiplier_cros}
                torch.save(checkpoint, train_model_path + '/checkpoint.pkl')
                torch.save(checkpoint, train_model_path + '/checkpoint_'+ str(curr_episode).zfill(6) +'.pkl')
                print('(Driver)Model Saved. version: checkpoint', end='\n')
            if best_reward is None:
                best_reward = ppo.driver_memory.recorder['total_reward']/ppo.driver_memory.recorder['total_step']
            elif ppo.driver_memory.recorder['total_reward']/ppo.driver_memory.recorder['total_step'] > best_reward:
                best_reward =  ppo.driver_memory.recorder['total_reward']/ppo.driver_memory.recorder['total_step']
                checkpoint = {'model_state_dict': weights,
                              'optimizer_state_dict': ppo.optimizer.state_dict(),
                              'epoch': curr_episode,
                              'lagrange_multiplier_lane':rcpo.lagrange_multiplier_lane,
                              'lagrange_multiplier_coll':rcpo.lagrange_multiplier_coll,
                              'lagrange_multiplier_cros':rcpo.lagrange_multiplier_cros}
                torch.save(checkpoint, train_model_path + '/checkpoint_best.pkl')
                print('(Driver)Model Saved. version best reward', end='\n')

            # clear memory
            ppo.driver_memory.clear_memory()
            rcpo.driver_memory.clear_memory()

            # new weight and new job
            weights = ppo.global_network.state_dict()
            # bug package: GLException(msg)
            if restart_counter < EPSIODE_PER_KILL:
                restart_counter += 1
                for i, meta_agent in enumerate(meta_agents):
                    job_list.append(meta_agent.job.remote(curr_episode, weights, rcpo))
                    curr_episode += 1
            else:
                restart_counter = 0
                for meta_agent in meta_agents:
                    ray.kill(meta_agent)
                    meta_agent.__del__.remote()
                rl_agents = [RLRunner.remote(i, rcpo) for i in range(NUM_RLRUNNER)]
                meta_agents = rl_agents
                job_list = []
                for i, meta_agent in enumerate(meta_agents):
                    job_list.append(meta_agent.job.remote(curr_episode, weights, rcpo))
                    curr_episode += 1
                print('(Driver)Meta restart meta at episode {0}'.format(curr_episode))

    except KeyboardInterrupt:
        print("CTRL-C pressed. Killing remote runners.")
        for meta_agent in meta_agents:
            ray.kill(meta_agent)


if __name__ == "__main__":
    main()
