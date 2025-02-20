import math
import ray
import threading
import numpy as np
import gym, gym_duckietown  # no change: registers the envs!

from Parameters import *

from Worker import Worker
from ACNet import ACNet
from Memory import *
from Tools import trans_state

from datetime import datetime


class Runner(object):
    """Actor object to start running simulation on workers.
        Gradient computation is also executed on this object."""

    def __init__(self, metaAgentID, rcpo):
        if SYS_DEBUG_MODE:
            print('((Runner)__init__) Start...')
        # basic info
        self.metaAgentID = metaAgentID
        self.num_workers = NUM_WORKER
        # driver info
        self.curr_episode = int(metaAgentID)
        # lagrange (asynchronous)
        self.driver_rcpo = rcpo
        self.lagrange_multiplier_lane = rcpo.lagrange_multiplier_lane
        self.lagrange_multiplier_coll = rcpo.lagrange_multiplier_coll
        self.lagrange_multiplier_cros = rcpo.lagrange_multiplier_cros

        # create net
        self.local_network = ACNet(ACTION_SIZE, RUNNER_DEVICE).to(RUNNER_DEVICE)
        self.local_network.share_memory()

        # create env
        self.env = gym.make(TOWN)
        self.env.seed(metaAgentID)
        # self.joint = {}
        # (do not change) BUG: 'pyglet.gl.lib.GLException'
        # self.env.reset()
        # self.env.reset() # where is this bug comes from? nothing changed. Just in Ray.
        # print('（(Runner)__init__) env reset')

        # create batch_memory
        self.runner_memory = BatchMemory('runner')
        if SYS_DEBUG_MODE:
            print('((Runner)__init__) Done!')

    def __del__(self):
        if SYS_DEBUG_MODE:
            print("((Runner)__del__) Done!")

    def reset(self, curr_episode, global_weights, rcpo):
        self.curr_episode = curr_episode
        self.set_weights(global_weights)
        self.set_lagrange(rcpo)
        self.runner_memory.clear_memory()

        self.env.close()
        self.env.reset()
    
    def set_lagrange(self, rcpo):
        self.driver_rcpo = rcpo
        self.lagrange_multiplier_lane = self.driver_rcpo.lagrange_multiplier_lane
        self.lagrange_multiplier_coll = self.driver_rcpo.lagrange_multiplier_coll
        self.lagrange_multiplier_cros = self.driver_rcpo.lagrange_multiplier_cros

    def get_weights(self):
        return self.local_network.state_dict()

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)
        self.local_network.eval()

    def single_rl(self):
        sub_memory = SubMemory()
        values = torch.tensor([0]).to(RUNNER_DEVICE)

        state = self.env.reset()
        if BOOL_RENDER:
            self.env.render()
        done = False
        lstm_states = None
        _lp = None

        # step begin
        for t in range(MAX_NUM_STEPS):
            # state and action
            states = trans_state(state)
            with torch.no_grad():# todo
                actions, values, lstm_states, action_logprobs, _ = \
                    self.local_network(states, lstm_state=lstm_states, old_action=None)
            action = actions.cpu().data.numpy().flatten()

            # step
            state, reward, done, info = self.env.step(action)

            # RCPO reward 
            def constrained_reward_function(info, done, _lp):
                try:
                    lp = info['Simulator']['lane_position']
                except:
                    # NotInLane
                    lp = None
                    reward = 0
                    cost_lane = LANE_FACTOR
                else:
                    if np.linalg.norm(info['Simulator']['delta_pos']):
                        delta_pos = info['Simulator']['delta_pos'] / np.linalg.norm(info['Simulator']['delta_pos'])
                        pose_angle = np.array([math.cos(info['Simulator']['cur_angle']), 0, -math.sin(info['Simulator']['cur_angle'])])
                        steer_dot = np.dot(pose_angle, delta_pos) / np.abs(np.dot(pose_angle, delta_pos))
                    else:
                        steer_dot = 0
                    reward = MOVE_REWARD * steer_dot * info['Simulator']['robot_speed'] * lp['dot_dir']
                    cost_lane = LANE_FACTOR * float(np.abs(lp['dist']))
                # cost_coll = 1 * abs(info['Simulator']['proximity_penalty'])
                cost_coll = 1 * 1 if done else 0

                # （xinwei - illegal）
                cost_cros = 0
                if _lp is not None and lp is not None:
                    lane_dis = float(np.abs(lp['dist']))
                    _lane_dis = float(np.abs(_lp['dist']))
                    if _lane_dis < 0.138 * SAFE_FACTOR and lane_dis > 0.138 * SAFE_FACTOR:
                        cost_cros = 1

                reward_hat = reward - self.lagrange_multiplier_lane * cost_lane - self.lagrange_multiplier_coll * cost_coll * COLLIDE_FACTOR - self.lagrange_multiplier_cros * cost_cros * CROSS_FACTOR
                # print('reward_hat', reward_hat, 'reward', reward,'cost_lane',cost_lane, 'cost_coll', cost_coll)
                reward_hat = REWARD_FACTOR * reward_hat
                reward = REWARD_FACTOR * reward
                return reward_hat, reward, cost_lane, cost_coll, cost_cros, lp
            reward_hat, reward, cost_lane, cost_coll, cost_cros, lp = constrained_reward_function(info, done, _lp)
            _lp = lp

            if BOOL_RENDER:
                self.env.render()

            # Record: state, action, logprob, reward, terminal
            # Input state with batch, shrink network output here.
            sub_memory.buffer['states'].append(np.squeeze(states.cpu().data.numpy(), axis=0))
            sub_memory.buffer['actions'].append(np.squeeze(actions.cpu().data.numpy(), axis=0))
            sub_memory.buffer['log_probs'].append(np.squeeze(action_logprobs.cpu().data.numpy(), axis=0))
            # convert MCR in update
            sub_memory.buffer['rewards'].append(reward_hat)
            sub_memory.buffer['return'].append(reward)
            sub_memory.buffer['is_terminals'].append(done)
            sub_memory.buffer['cost_lane'].append(cost_lane)
            sub_memory.buffer['cost_coll'].append(cost_coll)
            sub_memory.buffer['cost_cros'].append(cost_cros)

            # Recode: env info
            sub_memory.recorder['total_step'] += 1
            sub_memory.recorder['total_reward'] += reward
            sub_memory.recorder['total_done'] += 1 if done else 0
            sub_memory.recorder['robot_speed'] += info['Simulator']['robot_speed']
            if lp is not None:
                sub_memory.recorder['total_angle'] += lp['dot_dir']
                sub_memory.recorder['total_distance'] += \
                    info['Simulator']['robot_speed'] * lp['dot_dir'] * info['Simulator']['delta_time']
                sub_memory.recorder['total_lane'] += - LANE_FACTOR * np.abs(lp['dist'])
            sub_memory.recorder['total_reward_hat'] += reward_hat
            sub_memory.recorder['cost_lane'] += cost_lane
            sub_memory.recorder['cost_coll'] += cost_coll
            sub_memory.recorder['cost_cros'] += cost_cros

            if done:
                self.env.close()
                break

        if not done:
            states = trans_state(state)
            with torch.no_grad():
                _, values, _, _, _ = \
                        self.local_network(states, lstm_state=lstm_states, old_action=None)     
            sub_memory.buffer['rewards'][-1] += (GAMMA * (values.cpu().data.numpy()))
            print('boost', values.cpu().data.numpy())
            print('sub_memory.buffer["rewards"][-1]', sub_memory.buffer['rewards'][-1])

        # summarize
        sub_memory.buffer['meta'].append(self.metaAgentID)
        sub_memory.recorder['total_angle'] = sub_memory.recorder['total_angle'] / sub_memory.recorder['total_step']
        sub_memory.recorder['robot_speed'] = sub_memory.recorder['robot_speed'] / sub_memory.recorder['total_step']
        sub_memory.recorder['total_lane'] = sub_memory.recorder['total_lane'] / sub_memory.recorder['total_step']
        
        # drop memory if buffer is too small
        if sub_memory.recorder['total_step'] >= MIN_BUFFER_LENGTH:
            sub_memory.recorder['bool_learn'] = 1
        self.runner_memory.push_memory(sub_memory)
        print('{0} episode {1} runner {2} rl: {3}'.format(
            datetime.now().strftime("%m/%d %H:%M:%S"),
            self.curr_episode, self.metaAgentID, 
            {key : round(self.runner_memory.recorder[key], 2) for key in self.runner_memory.recorder}))
        
    def job(self, curr_episode, global_weights, rcpo):
        info = {"id": self.metaAgentID,
                "episode_number": curr_episode}
        self.reset(curr_episode, global_weights, rcpo)

        if SYS_DEBUG_MODE:
            print("(job)episode {0} on metaAgent {1} reinforcement...".format(curr_episode, self.metaAgentID))

        if SINGLE_DEBUG_MODE:
            self.single_rl()
        else:
            self.threads_rl()
            if EPSIODE_PER_KILL > 5:
                raise ValueError('EPSIODE_PER_KILL: Lager than 5. 4 recommend.')
            raise NotImplementedError

        return info, self.runner_memory

@ray.remote(num_cpus=CPU_PER_RUNNER, num_gpus=GPU_PER_RUNNER)
class RLRunner(Runner):
    def __init__(self, metaAgentID, lagrange_target_lane):
        super().__init__(metaAgentID, lagrange_target_lane)
