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
