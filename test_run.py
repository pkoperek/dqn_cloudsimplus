# code based on
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import gym_cloudsimplus
import os
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make('SingleDCAppEnv-v0').unwrapped

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


storage_path = os.getenv('STORAGE_PATH', '/storage')


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(5, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 3)

    def forward(self, x):
        print("Network input: " + str(x.size()))
        x = F.selu(self.bn1(self.conv1(x)))
        x = F.selu(self.bn2(self.conv2(x)))
        x = F.selu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def get_measurements():
    measurements = env.render(mode='array')
    measurements = torch.tensor(measurements)
    measurements = measurements.unsqueeze(0)

    return measurements


env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())

steps_done = 0

policy_net = torch.load(storage_path + '/policy_net_full.dump')
target_net = torch.load(storage_path + '/target_net_full.dump')


def select_action(state):
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


episode_durations = []

total_reward = 0
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
    _, reward, done, _ = env.step(action.item())
    total_reward += reward
    print("Reward for action: " + str(reward) + " act: " + str(action))
    reward = torch.tensor([reward], device=device)

    # Observe new state
    last_measurements = current_measurements
    current_measurements = get_measurements()

    number_of_vms = current_measurements[0][0][-1].item()
    a = action[0][0].item()

    print("LOG it vm act: {} {} {}".format(
        str(t),
        str(number_of_vms),
        str(a),
    ))

    if not done:
        next_state = current_measurements - last_measurements
    else:
        next_state = None

    # Move to the next state
    state = next_state

    print("Total reward: " + str(total_reward))

print('Complete')
env.close()
