import numpy as np
from Parameters import *
import copy
from Memory import SubMemory
from Tools import trans_state
import gym, gym_duckietown  # (do not change) registers the envs!


class Worker:
    def __init__(self, metaAgentID, workerID, num_workers, currEpisode,
                 env, local_network, group_lock, joint):
        # Worker information
        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_agents = num_workers
        self.currEpisode = currEpisode

        self.env = env
        self.local_network = local_network
        self.group_lock = group_lock

        self.sub_memory = SubMemory()

    def __del__(self):
        if SYS_DEBUG_MODE and self.agentID == 1:
            print("((worker)__del__) Done!")

    def single_work(self):
        if SYS_DEBUG_MODE:
            print('(Worker-RL)Begin to run! meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))
        bool_result = True

        state = self.env.reset()

        if BOOL_RENDER:     self.env.render()
        lstm_states = None

        # step begin
        for t in range(MAX_NUM_STEPS):
            # state and action
            states = trans_state(state)
            actions, values, lstm_states, action_logprobs, dist_entropys = \
                self.local_network(states, lstm_state=lstm_states, old_action=None)
            action = actions.cpu().data.numpy().flatten()
            def return_vel_steer(u_r,u_l):
                '''baseline = self.unwrapped.wheel_dist
                # assuming same motor constants k for both motors
                k_r = self.k
                k_l = self.k

                # adjusting k by gain and trim
                k_r_inv = (self.gain + self.trim) / k_r
                k_l_inv = (self.gain - self.trim) / k_l

                omega_r = (vel + 0.5 * angle * baseline) / self.radius
                omega_l = (vel - 0.5 * angle * baseline) / self.radius 

                # conversion from motor rotation rate to duty cycle
                u_r = omega_r * k_r_inv
                u_l = omega_l * k_l_inv 
                '''
                k_r = K
                k_l = K
                k_r_inv = (GAIN + TRIM)/k_r
                k_l_inv = (GAIN - TRIM)/k_l
                w_r  = u_r/k_r_inv
                w_l  = u_l/k_l_inv
                b    = WHEEL_DIST
                vel   = (w_r + w_l)*RADIUS/2
                angle = (w_r - w_l)*RADIUS/b
                return np.array([vel,angle])
           # print(action)
            if action[0] == 0:
                action = return_vel_steer(0.4,0.04)
            if action[0] == 1:
                action = return_vel_steer(0.04,0.4)
            if action[0] == 2:
                action = return_vel_steer(0.3,0.3)

            # step
            state, reward, done, _ = self.env.step(action)
            if BOOL_RENDER:     self.env.render()

            # Record: state, action, logprob, reward, terminal
            # Input state with batch, shrink network output here.
            self.sub_memory.buffer['states'].append(np.squeeze(states.cpu().data.numpy(), axis=0))
            self.sub_memory.buffer['actions'].append(np.squeeze(actions.cpu().data.numpy(), axis=0))
            self.sub_memory.buffer['log_probs'].append(np.squeeze(action_logprobs.cpu().data.numpy(), axis=0))
            # convert MCR in update
            self.sub_memory.buffer['rewards'].append(reward)  # if not done else -1000
            self.sub_memory.buffer['is_terminals'].append(done)

            # Recode: env info
            self.sub_memory.recorder['total_step'] += 1
            self.sub_memory.recorder['total_reward'] += reward
            self.sub_memory.recorder['total_done'] += 1 if done else 0

            # without if update
            if done:
                # self.env.close()
                break

        # drop memory if buffer is too small
        if self.sub_memory.recorder['total_step'] > MIN_BUFFER_LENGTH:
            self.sub_memory.recorder['bool_learn'] = 1

    def work(self):
        if self.num_agents == 1:
            self.single_work()
        else:
            raise NotImplementedError

    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.group_lock.release(int(self.lock_bool), self.name)
        self.group_lock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool
