import gym
import gym_cloudsimplus

env = gym.make('SingleDCAppEnv-v0')
env.reset()
observation, reward, done, info = env.step(1)
