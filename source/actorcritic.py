# -*- coding: utf-8 -*-

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4,256)
        self.fc2 = nn.Linear(256,2)
        self.fc3 = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
    #two NNs
    def pi(self, x, softmax_dim = 0): # pi NN
        pi_x = self.fc2(F.relu(self.fc1(x)))
        return F.softmax(pi_x, dim=softmax_dim) # softmax
    def v(self, x): # value NN
        v = self.fc3(F.relu(self.fc1(x))) # No need for softmax because value is not probabaility
        return v

    def makebatch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            self.data = []
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_lst, dtype=torch.float)
    #train
    def train(self):
        s, a, r, s_prime, done = self.makebatch() # Create tensor using data entered through makebatch
        td_target = r + 0.98 * self.v(s_prime) * done # td_target: Correct answer, 0.98 is the gamma value
        delta = td_target - self.v(s) # target - v(s)
        pi = self.pi(s, softmax_dim=1) # pi action probabilities
        pi_a = pi.gather(1,a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())
              #policy_loss                         # value_loss
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

env = gym.make('CartPole-v1')
model = ActorCritic()
print_interval = 20
score = 0.0
for n_epi in range(10000):
    done = False
    s = env.reset()
    while not done:
        for t in range (10): # n-step and update
            prob = model.pi(torch.from_numpy(s).float()) # Take action with policy
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, r, done, info = env.step(a)
            model.data.append((s,a,r,s_prime,done))
            s = s_prime
            score += r
            if done:
                break
        model.train()
    if n_epi%print_interval==0 and n_epi!=0:
        print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
        score = 0.0
env.close()