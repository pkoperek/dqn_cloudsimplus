import gym
import gym_cloudsimplus
import math
import random
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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv1d(5, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 2)

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
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]],
                            device=device,
                            dtype=torch.long)


episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.uint8
    )
    non_final_next_states = torch.cat([
        s for s in batch.next_state
        if s is not None
    ])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    print("Episode: " + str(i_episode))

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

        print("LOG e it vm act: {} {} {} {}".format(
            str(i_episode),
            str(t),
            str(number_of_vms),
            str(a),
        ))

        if not done:
            next_state = current_measurements - last_measurements
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            print("Total reward: " + str(total_reward))
            break
    # Update the target network
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.close()
