import argparse
import gym
import numpy as np
from itertools import count
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from model import Policy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G')
    parser.add_argument('--seed', type=int, default=543, metavar='N')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N')
    return parser.parse_args()


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.policy = Policy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.env = gym.make('CartPole-v0')
        self.env.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(Variable(state))
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.data[0]

    def finish_eposide(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.args.gamma * R
            rewards.insert(0, R)
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) /\
            (rewards.std() + np.finfo(np.float32).eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(- log_prob * reward)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]

    def train(self):
        running_reward = 10
        for i_episode in count(1):
            t = self.train_one_episode()
            running_reward = running_reward * 0.99 + t * 0.01
            self.finish_eposide()
            if i_episode % self.args.log_interval == 0:
                print('episode: {}th \t current len: {:5d} \t ave \
                      len: {:.2f}'.format(
                      i_episode, t, running_reward))
            if running_reward > self.env.spec.reward_threshold:
                print('Solved,',
                      'Runnig reward is now {} and'.format(running_reward),
                      'the last episode runs to {} time steps'.format(t))
                break

    def train_one_episode(self):
        state = self.env.reset()
        for t in range(10000):
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            if self.args.render:
                self.env.render()
            self.policy.rewards.append(reward)
            if done:
                break
        return t

    def demo(self):
        state = self.env.reset()
        for t in range(10000):
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            self.env.render()
            if done:
                break


if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.demo()
