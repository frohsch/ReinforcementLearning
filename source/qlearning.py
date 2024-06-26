# -*- coding: utf-8 -*-

import numpy as np
import random
Q = np.zeros([16,4])
num_episodes = 1000

def rargmax(vector):
  m = np.amax(vector)
  indices = np.nonzero(vector == m)[0]
  return random.choice(indices)

def RL_step(state,action):
  # Computation: New State
  if action == 0: # LEFT
    if state % 4 == 0:
      new_state = state
    else:
      new_state = state - 1
  elif action == 1: # DOWN
    if state > 11:
      new_state = state
    else:
      new_state = state + 4
  elif action == 2: # RIGHT
    if (state+1)%4 == 0:
      new_state = state
    else:
      new_state = state + 1
  else: # UP
    if state < 4:
      new_state = state
    else:
      new_state = state - 4
  # Computation: Reward
  if new_state == 15:
    reward = 1
  elif (new_state == 5) or (new_state == 7) or (new_state == 11) or (new_state == 12):
    new_state = state
    reward = 0
  else:
    reward = 0
  # Computation: Done
  if new_state == 15:
    done = True
  else:
    done = None
  return new_state, reward, done

for i in range(num_episodes):
  state = 0 # 시작점
  reward = 0 # 누적된 리워드값의 초기값
  done = None # done이라는 변수는 끝났는지 여부를 Binary로 말해줌.

  while not done:
    action = rargmax(Q[state,:])
    new_state, reward, done = RL_step(state, action)
    Q[state, action] = reward + np.max(Q[new_state,:])
    state = new_state
print(Q)