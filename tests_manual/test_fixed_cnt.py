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

total_reward = 0
total_wait = 0
# Initialize the environment and state - restart the simulation
env.reset()

# Run the simulation until we are 'done'
for t in count():
    print("Iteration: " + str(t))
    # Select and perform an action
    obs, reward, done, _ = env.step(0)
    total_reward += reward
    total_wait += obs[5]
    print("Reward for action: " + str(reward) +
          " act: 0" +
          " total reward: " + str(total_reward) +
          " total wait: " + str(total_wait) +
          " obs: " + str(list(obs)))

    if done:
        break

print('Complete')
print("Total reward: " + str(total_reward) + " total wait time: " + str(total_wait))

env.close()
