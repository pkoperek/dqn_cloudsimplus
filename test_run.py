# code based on
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import gym_cloudsimplus
import os
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('SingleDCAppEnv-v0').unwrapped

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 3)

    def forward(self, x):
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.selu(self.bn2(self.conv2(x)))
        x = F.selu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_measurements():
    measurements = env.render(mode='array')
    measurements = torch.tensor(measurements)
    measurements = measurements.unsqueeze(0)

    return measurements


def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


storage_path = os.getenv('STORAGE_PATH', '/storage')
loaded_state_dict = torch.load(storage_path + '/policy_net_state.dump')
policy_net = DQN().to(device)
policy_net.load_state_dict(loaded_state_dict)

env.reset()

total_reward = 0
total_wait = 0
# Initialize the environment and state - restart the simulation
env.reset()
last_measurements = get_measurements()
current_measurements = get_measurements()
state = current_measurements - last_measurements

# Run the simulation until we are 'done'
for t in count():
    print("Iteration: " + str(t))
    # Select and perform an action
    action = select_action(state)
    obs, reward, done, _ = env.step(action.item())
    total_reward += reward
    total_wait += obs[5]
    print("Reward for action: " + str(reward) +
          " act: " + str(action) +
          " total reward: " + str(total_reward) +
          " total wait: " + str(total_wait) +
          " obs: " + str(list(obs)))

    # Observe new state
    last_measurements = current_measurements
    current_measurements = get_measurements()

    if not done:
        next_state = current_measurements - last_measurements
    else:
        break

    # Move to the next state
    state = next_state

print('Complete')
print("Total reward: " + str(total_reward) + " total wait time: " + str(total_wait))

env.close()
