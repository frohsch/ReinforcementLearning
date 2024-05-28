# -*- coding: utf-8 -*-

import numpy as np
import gym
env = gym.make('FrozenLake-v1', is_slippery=False)
env = env.unwrapped # enwrap it to have additional information from it
V = np.zeros(env.observation_space.n)
policy = np.zeros(env.observation_space.n)
policy_stable = False
i = 0
eps = 0.0001
tot_rew = 0

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (r + gamma*V[next_s]) for p, next_s, r, _ in env.P[s][a]])

while not policy_stable:
    #policy evaluation
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            old_v = V[s]
            V[s] = eval_state_action(V, s, policy[s]) # update V[s] using the Bellman equation
            delta = max(delta, np.abs(old_v - V[s]))
        if delta < eps:
            break
    #policy improvement
    policy_stable = True
    for s in range(env.observation_space.n):
        old_a = policy[s]
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(env.action_space.n)])
        if old_a != policy[s]:
            policy_stable = False
    i += 1
print('Converged after %i policy iterations'%(i))

state = env.reset()
n_epi = 0
for _ in range(100):
    done = False
    while not done:
        next_state, reward, done, _, _ = env.step(policy[state])
        state = next_state
        tot_rew += reward
        if done:
            state = env.reset()
    n_epi += 1
print('Won %i of 100 games!'%(tot_rew))
print('V values: \n', V.reshape((4,4)), '\nOptimal policy: \n', policy.reshape((4,4)))