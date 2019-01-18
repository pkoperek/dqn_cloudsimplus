# code based on
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import gym
import gym_cloudsimplus
import math
import os
import random
from collections import namedtuple
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging

log = logging.getLogger('infinity.learn')
log.setLevel(logging.DEBUG)

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
        self.conv1 = nn.Conv1d(6, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.head = nn.Linear(32 * 222, 3)

    def forward(self, x):
        log.debug("Network input: " + str(x.size()))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class ModelTrainer(object):

    def __init__(self):
        super(ModelTrainer, self).__init__()

        self._storage_path = os.getenv('STORAGE_PATH', '/storage')
        self._episodes_cnt = int(os.getenv('EPISODES_CNT', '1'))
        self._device = os.getenv('DEVICE', 'cpu')
        self._policy_net = DQN().to(self._device)
        self._target_net = DQN().to(self._device)
        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()
        self._optimizer = optim.RMSprop(self._policy_net.parameters())
        self._memory = ReplayMemory(10000)
        self._batch_size = int(os.getenv('BATCH_SIZE', '128'))
        self._gamma = 0.999
        self._eps_start = 0.9
        self._eps_end = 0.05
        self._eps_delay = 200
        self._target_update = 10

    @property
    def episode_durations(self):
        return self._episode_durations

    def _get_measurements(self):
        measurements = self._env.render(mode='array')
        measurements = torch.tensor(measurements)
        measurements = measurements.unsqueeze(0)

        return measurements

    def _select_action(self, state):
        sample = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * \
            math.exp(-1. * self._steps_done / self._eps_delay)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self._policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(3)]],
                                device=self._device,
                                dtype=torch.long)

    def _optimize_model(self):
        if len(self._memory) < self._batch_size:
            # we want to have at least BATCH_SIZE elements in the memory
            return

        transitions = self._memory.sample(self._batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self._device,
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
        states = self._policy_net(state_batch)

        log.debug("States size: " + str(states.size()))

        state_action_values = states.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self._batch_size, device=self._device)
        next_state_values[non_final_mask] =\
            self._target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values =\
            (next_state_values * self._gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        for param in self._policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def train(self, simulation_start, simulation_end):
        self._steps_done = 0
        self._episode_durations = []
        self._env = gym.make('SingleDCAppEnv-v0').unwrapped
        max_total_reward = None
        for i_episode in range(self._episodes_cnt):
            log.debug("Episode: " + str(i_episode))

            total_reward = 0
            # Initialize the environment and state - restart the simulation
            self._env.reset(simulation_start, simulation_end)
            last_measurements = self._get_measurements()
            current_measurements = self._get_measurements()
            state = current_measurements - last_measurements

            log.debug(f'Input size: {state.size()}')
            log.debug('Training: ' + str(simulation_start) + ' till ' +
                      str(simulation_end))

            # run the simulation for a constant amount of time
            # ... or until it is 'done' - the status returned is 'done'
            start = int(simulation_start / 1000)
            end = int(simulation_end / 1000) + 1

            log.debug('Running the episode for: ' + str(end-start+1) + 's')
            for t in range(end - start + 1):
                log.debug("Iteration: " + str(t))
                # Select and perform an action
                action = self._select_action(state)
                _, reward, done, _ = self._env.step(action.item())
                total_reward += reward
                log.debug(
                    "Reward for action: "
                    + str(reward)
                    + " act: "
                    + str(action)
                )
                reward = torch.tensor([reward], device=self._device)

                # Observe new state
                last_measurements = current_measurements
                current_measurements = self._get_measurements()

                number_of_vms = current_measurements[0][0][-1].item()
                a = action[0][0].item()

                log.debug(f"LOG e it vm act: {i_episode} {t} {number_of_vms} {a}")

                if not done:
                    next_state = current_measurements - last_measurements
                else:
                    next_state = None

                # Store the transition in memory
                self._memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self._optimize_model()
                if done:
                    self._episode_durations.append(t + 1)
                    log.debug("Total reward: " + str(total_reward))
                    if max_total_reward:
                        if max_total_reward < total_reward:
                            max_total_reward = total_reward
                    else:
                        max_total_reward = total_reward
                    break
            # Update the target network
            if i_episode % self._target_update == 0:
                self._target_net.load_state_dict(
                    self._policy_net.state_dict()
                )

        log.debug('Complete')

        self._env.close()

        log.debug('Saving the result of training')

        buff = BytesIO()
        torch.save(self._policy_net.state_dict(), buff)
        buff.seek(0)
        return buff, max_total_reward
