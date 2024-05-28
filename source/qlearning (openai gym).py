# -*- coding: utf-8 -*-

import numpy as np
import gym
import random
env = gym.make('FrozenLake-v1', is_slippery=False)
env = env.unwrapped # enwrap it to have additional information from it
Q = np.zeros([env.observation_space.n,env.action_space.n])
num_episodes = 1000

def rargmax(vector):
  m = np.amax(vector)
  indices = np.nonzero(vector == m)[0]
  return random.choice(indices)

for _ in range(num_episodes):
  state = env.reset()
  reward = 0
  done = None
  while not done:
    action = rargmax(Q[state,:])
    new_state, reward, done, _, _  = env.step(action)
    Q[state, action] = reward + np.max(Q[new_state, :])
    state = new_state
print(Q)