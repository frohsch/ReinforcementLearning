# -*- coding: utf-8 -*-

import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
gamma        = 0.99
batch_size   = 32
buffer_limit = 50000
tau          = 0.005

class ReplayBuffer():
  def __init__(self):
    self.buffer = collections.deque(maxlen=buffer_limit)
  def put(self, transition):
    self.buffer.append(transition)
  def sample(self, n):
    mini_batch = random.sample(self.buffer, n)
    s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
    for transition in mini_batch:
      s, a, r, s_prime, done = transition
      s_lst.append(s)
      a_lst.append([a])
      r_lst.append([r])
      s_prime_lst.append(s_prime)
      done_mask = 0.0 if done else 1.0
      done_mask_lst.append([done_mask])
      return_s = torch.tensor(s_lst, dtype=torch.float)
      return_a = torch.tensor(a_lst, dtype=torch.float)
      return_r = torch.tensor(r_lst, dtype=torch.float)
      return_s_prime = torch.tensor(s_prime_lst, dtype=torch.float)
      return_done = torch.tensor(done_mask_lst, dtype=torch.float)
    return return_s, return_a, return_r, return_s_prime, return_done
  def size(self):
    return len(self.buffer)

class Actor(nn.Module):
  def __init__(self):
    super(Actor, self).__init__()
    self.fc1 = nn.Linear(3, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc_mu = nn.Linear(64, 1)
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
    return mu

class Critic(nn.Module):
  def __init__(self):
    super(Critic, self).__init__()
    self.fc_s = nn.Linear(3, 64)
    self.fc_a = nn.Linear(1,64)
    self.fc_q = nn.Linear(128, 32)
    self.fc_out = nn.Linear(32,1)
  def forward(self, x, a):
    h1 = F.relu(self.fc_s(x))
    h2 = F.relu(self.fc_a(a))
    cat = torch.cat([h1,h2], dim=1)
    q = F.relu(self.fc_q(cat))
    q = self.fc_out(q)
    return q

class OrnsteinUhlenbeckNoise:
  def __init__(self, actor):
    self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
    self.actor = actor
    self.x_prev = np.zeros_like(self.actor)
  def __call__(self):
    x = self.x_prev + self.theta * (self.actor - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.actor.shape)
    self.x_prev = x
    return x

def train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer):
  s,a,r,s_prime,done_mask  = memory.sample(batch_size)
  target = r + gamma * critic_target(s_prime, actor_target(s_prime)) * done_mask

  critic_loss = F.smooth_l1_loss(critic(s,a), target.detach())
  critic_optimizer.zero_grad()
  critic_loss.backward()
  critic_optimizer.step()

  actor_loss = -critic(s,actor(s)).mean()
  actor_optimizer.zero_grad()
  actor_loss.backward()
  actor_optimizer.step()

def soft_update(net, net_target):
  for param_target, param in zip(net_target.parameters(), net.parameters()):
    param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

env = gym.make('Pendulum-v1', max_episode_steps=200, autoreset=True)
memory = ReplayBuffer()

critic, critic_target = Critic(), Critic()
critic_target.load_state_dict(critic.state_dict())
actor, actor_target = Actor(), Actor()
actor_target.load_state_dict(actor.state_dict())

score = 0.0
actor_optimizer = optim.Adam(actor.parameters(), lr=0.0005)
critic_optimizer  = optim.Adam(critic.parameters(), lr=0.001)
ou_noise = OrnsteinUhlenbeckNoise(actor=np.zeros(1))

for n_epi in range(10000):
  s, _ = env.reset()
  done = False
  count = 0
  while count < 200 and not done:
    a = actor(torch.from_numpy(s).float())
    a = a.item() + ou_noise()[0]
    s_prime, r, done, truncated, info = env.step([a])
    memory.put((s,a,r/100.0,s_prime,done))
    score +=r
    s = s_prime
    count += 1
  if memory.size()>2000:
    for i in range(10):
      train(actor, actor_target, critic, critic_target, memory, critic_optimizer, actor_optimizer)
      soft_update(actor, actor_target)
      soft_update(critic,  critic_target)
  if n_epi%50==0 and n_epi!=0:
    print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/50))
    score = 0.0
env.close()



