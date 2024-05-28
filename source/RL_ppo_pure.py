import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class RL(nn.Module):
  def __init__(self):
    super(RL, self).__init__()
    self.data = []
    self.hl_1  = nn.Linear(4, 256)
    self.nn_pi = nn.Linear(256, 2)
    self.nn_v  = nn.Linear(256, 1)
    self.optimizer = optim.Adam(self.parameters(), lr = 0.0002)
  # pi (actor)
  def pi(self, x, softmax_dim=0):
    pi_x = self.nn_pi(F.relu(self.hl_1(x)))
    return F.softmax(pi_x, dim=softmax_dim)
  # v (critic)
  def v(self, x):
    return self.nn_v(F.relu(self.hl_1(x)))
  def make_batch(self):
    list_s, list_a, list_r, list_s_prime, list_prob_a, list_done = [], [], [], [], [], []
    for transition in self.data:
      s, a, r, s_prime, prob_a, done = transition
      list_s.append(s)
      list_a.append([a])
      list_r.append([r/100.0])
      list_s_prime.append(s_prime)
      list_prob_a.append([prob_a])
      mask_done = 0.0 if done else 1.0
      list_done.append([mask_done])
      s = torch.tensor(list_s, dtype=torch.float)
      a = torch.tensor(list_a)
      r = torch.tensor(list_r, dtype=torch.float)
      s_prime = torch.tensor(list_s_prime, dtype=torch.float)
      prob_a = torch.tensor(list_prob_a, dtype=torch.float)
      done = torch.tensor(list_done, dtype=torch.float)
      self.data = []
    return s, a, r, s_prime, prob_a, done
  def train(self):
    s, a, r, s_prime, prob_a, done = self.make_batch()
    eps_clip = 0.1
    # [ PPO 기반 학습 ]================================
    td_target = r + 0.98 * self.v(s_prime) * done
    delta = td_target - self.v(s)
    delta = delta.detach().numpy()
    list_advantage = []
    advantage = 0.0
    for delta_t in delta[::-1]:
      advantage = 0.98 + 0.95 * advantage + delta_t[0]
      list_advantage.append([advantage])
    list_advantage.reverse()
    advantage = torch.tensor(list_advantage, dtype=torch.float)
    pi = self.pi(s, softmax_dim=1)
    pi_a = pi.gather(1,a)
    ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())
    # =================================================
    self.optimizer.zero_grad()
    loss.mean().backward()
    self.optimizer.step()

env = gym.make('CartPole-v1')
model = RL()
score = 0.0
for n_epi in range(10000):
  done = False
  s = env.reset()
  while not done:
    for t in range(20):
      prob = model.pi(torch.from_numpy(s).float())
      m = Categorical(prob)
      a = m.sample().item()
      s_prime, r, done, info = env.step(a)
      model.data.append([s, a, r, s_prime, prob[a].item(), done])
      s = s_prime
      score += r
      if done:
        break
    model.train()
  if n_epi%100==0 and n_epi!=0:
    print("# of episode: {}, avg score: {:.1f}".format(n_epi, score/50))
    score = 0.0
env.close()