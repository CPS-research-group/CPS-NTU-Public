import torch
import torch.nn as nn
from ACNet import ACNet
from Parameters import *
from Memory import *

torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)

def normalize_rewards(rewards):
    return (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)

# PPO method
class PPO:
    def __init__(self, action_dim, device):
        # train parameter
        self.lr = LEARNING_RATE
        self.betas = BETAS
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCH_PPO
        self.device = device

        # AC net
        self.global_network = ACNet(ACTION_SIZE, self.device).to(self.device)
        self.global_network.share_memory()
        self.optimizer = torch.optim.Adam(self.global_network.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

        # batch batch_memory
        self.driver_memory = BatchMemory('driver')

    def get_weights(self):
        return self.global_network.state_dict()

    def set_weights(self, weights):
        self.global_network.load_state_dict(weights)

    # driver buffer
    def driver_update(self):
        """
        This is the update for driver buffer.
        :return:
        """
        # recorder
        start_loss = loss_recorder.copy()
        end_loss = loss_recorder.copy()
        gradiant = torch.tensor(0.).to(self.device)
        weight_variance = 0
        grad_norms = 0

        num_batch = len(self.driver_memory.batch_buffer)

        # Optimize policy for K epochs:
        for epoch in range(self.k_epochs):
            sum_loss = torch.tensor(0.).to(self.device)

            for batch_index in range(num_batch):
                worker_memory_buffer = self.driver_memory.batch_buffer[batch_index]
                # reward
                H = len(worker_memory_buffer['cost_lane'])
                rewards = []
                discounted_reward = 0
                for reward, is_terminal in zip(reversed(worker_memory_buffer['rewards']),
                                               reversed(worker_memory_buffer['is_terminals'])):
                    discounted_reward = reward + int(not is_terminal) * self.gamma * discounted_reward
                    rewards.insert(0, discounted_reward)
                cost_lanes = []
                average_cost = 0
                for cost, is_terminal in zip(reversed(worker_memory_buffer['cost_lane']),
                                               reversed(worker_memory_buffer['is_terminals'])):
                    average_cost = cost + int(not is_terminal) * average_cost
                    cost_lanes.insert(0, average_cost)
                cost_colls = []
                average_cost = 0
                for reward, is_terminal in zip(reversed(worker_memory_buffer['cost_coll']),
                                               reversed(worker_memory_buffer['is_terminals'])):
                    average_cost = cost/H + int(not is_terminal) * average_cost
                    cost_colls.insert(0, average_cost)
                cost_cros = []
                average_cost = 0
                for reward, is_terminal in zip(reversed(worker_memory_buffer['cost_cros']),
                                               reversed(worker_memory_buffer['is_terminals'])):
                    average_cost = cost/H + int(not is_terminal) * average_cost
                    cost_cros.insert(0, average_cost)
                # print('point 3', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())

                # Normalizing the rewards:
                rewards = normalize_rewards(np.array(rewards)+np.array(cost_lanes)+np.array(cost_colls)+np.array(cost_cros))
                rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device).detach()

                # Convert list to tensor
                old_states = torch.Tensor(np.array(worker_memory_buffer['states'])).to(self.device).detach()
                old_actions = torch.Tensor(np.array(worker_memory_buffer['actions'])).to(self.device).detach()
                old_logprobs = torch.Tensor(np.array((worker_memory_buffer['log_probs']))).to(self.device).detach()
                '''update tensor check point'''
                if NN_DEBUG_MODE:
                    print('env reward', worker_memory_buffer['rewards'])
                    print('Monte Carlo Rewards:', rewards)

                # Evaluating old actions and values :
                try:  
                    _, state_values, _, logprobs, dist_entropy = \
                        self.global_network(old_states, lstm_state=None, old_action=old_actions)
                except:
                    # error when memory is not correct
                    print(old_states, 'old_actions', old_actions)
                    print('buffer', worker_memory_buffer)
                    print('memory_allocated', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                    continue

                ratios = torch.exp(logprobs - old_logprobs)
                if NN_DEBUG_MODE:
                    print('clamp(ratios)', torch.clamp(ratios, 1. - self.eps_clip, 1. + self.eps_clip))

                # Finding Surrogate Loss:
                advantages = rewards - state_values
                surr1 = ratios * advantages.detach()
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.detach()
                # actor_loss
                actor_loss = -1. * torch.mean(torch.min(surr1, surr2))
                # critic_loss
                critic_loss = .5 * torch.mean(self.MseLoss(state_values, rewards))/(CITIC_NET_FACTOR*CITIC_NET_FACTOR)
                if NN_DEBUG_MODE:
                    print('state_values', state_values)
                    print('rewards', rewards)
                    print('state_values - rewards', state_values - rewards)
                    print('torch.square', torch.square(state_values - rewards))
                    print('torch.mean', torch.mean(torch.square(state_values - rewards)))
                    print('critic_loss', critic_loss)
                    print('actor_loss', actor_loss)
                # entropy_loss
                entropy_loss = - ENTROPY_FACTOR * torch.mean(dist_entropy)
                # total loss
                loss = actor_loss + critic_loss + entropy_loss
                if NN_DEBUG_MODE:
                    print('mean_loss:%s  actor_loss:%s  critic_loss:%s  entropy_loss:%s' % (torch.mean(loss),
                                                                                            torch.mean(actor_loss),
                                                                                            torch.mean(critic_loss),
                                                                                            torch.mean(entropy_loss)))

                if epoch == 0 and batch_index == 0:
                    start_loss['total_loss'] = float(torch.mean(loss).detach().cpu().numpy())
                    start_loss['actor_loss'] = float(torch.mean(actor_loss).detach().cpu().numpy())
                    start_loss['critic_loss'] = float(torch.mean(critic_loss).detach().cpu().numpy())
                    start_loss['entropy_loss'] = float(torch.mean(entropy_loss).detach().cpu().numpy())
                if epoch == (self.k_epochs - 1) and batch_index == (len(self.driver_memory.batch_buffer) - 1):
                    end_loss['total_loss'] = float(torch.mean(loss).detach().cpu().numpy())
                    end_loss['actor_loss'] = float(torch.mean(actor_loss).detach().cpu().numpy())
                    end_loss['critic_loss'] = float(torch.mean(critic_loss).detach().cpu().numpy())
                    end_loss['entropy_loss'] = float(torch.mean(entropy_loss).detach().cpu().numpy())

                # release gpu memory, delete rewards.
                del old_states, old_actions, old_logprobs, rewards

                self.optimizer.zero_grad()
                try:
                    loss.backward()
                except:
                    # error when gradiant blow out
                    print('memory_allocated', torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                gradiant = torch.nn.utils.clip_grad_norm_(self.global_network.parameters(), GRIDIANT_CLIP)
                self.optimizer.step()
                self.optimizer.zero_grad()
                del loss, actor_loss, critic_loss, entropy_loss, dist_entropy, state_values, surr1, surr2, advantages, ratios
                
                if epoch == 0 and batch_index == 0:
                    # record the weight_variance
                    weight_variance = torch.linalg.norm(torch.stack(
                        [torch.linalg.norm(p.detach()) for p in self.global_network.parameters()])).to(TORCH_CPU).detach().numpy()
                    grad_norms = torch.mean(gradiant).to(TORCH_CPU).detach().numpy()

        update_result = (start_loss, end_loss, weight_variance, grad_norms)
        return update_result
