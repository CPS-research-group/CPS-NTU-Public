import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from Parameters import *

torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ACNet(nn.Module):
    def __init__(self, action_dim, device):
        super(ACNet, self).__init__()
        self.device = device
        # conv
        self.conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(CHANNAL, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(NET_SIZE // 8, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(NET_SIZE // 4, NET_SIZE // 2, kernel_size=2, stride=1))

        # shared full
        self.shared_full = nn.Sequential(
            nn.Dropout(DROP_PROB),
            nn.Linear(6912, NET_SIZE),
            nn.Sigmoid())

        # LSTM layer
        # self.lstm = nn.LSTM(NET_SIZE, NET_SIZE, num_layers=1, batch_first=True)

        # actor
        self.actor_full = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE))
        self.actor_mu = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, action_dim))
        # critic
        self.critic_full = nn.Sequential(
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, NET_SIZE),
            nn.Linear(NET_SIZE, 1))

        self.apply(weights_init)

    def forward(self, state, lstm_state=None, old_action=None):
        # share full
        conv = self.conv(state)
        shared_full = self.shared_full(conv.view(conv.size(0), -1))
        shared_lstm = shared_full
        # actor
        output_network = ACTOR_MEAN_FACTOR * nn.functional.softmax(self.actor_mu(self.actor_full(shared_lstm))/ACTOR_MEAN_FACTOR) #Gangadhar modified here
        # critic
        state_value = CITIC_NET_FACTOR * self.critic_full(shared_lstm)

        try:
            output_network = output_network
            dist = Categorical(output_network)
        except:
            # error when gradiant blow out
            print('state', state, 'state_value', state_value, 'action_mean', action_mean, 'cov_mat', cov_mat)
            print(dist)
            raise ValueError
        if old_action is None:
            action = dist.sample()
            if  BOOL_TRAINING == False:
               action = (torch.argmax(output_network.flatten()).reshape(1,)).to(self.device)
        else:
            action = old_action
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action.detach(), torch.squeeze(state_value), lstm_state, action_logprob, dist_entropy


if __name__ == '__main__':
    import time
    device = torch.device("cuda:0")
    torch.set_default_tensor_type(torch.DoubleTensor)
    ac = [ACNet(2, device).to(device) for _ in range(1)]
    rnn_state = None
    state0 = None
    print(ac[0])
    test_tensor = 255 * np.random.random(size=(1, 3, 30, 40))
    print('test_tensor', test_tensor)
    print(ac[0](torch.tensor(test_tensor).to(device)))
    time.sleep(60)
