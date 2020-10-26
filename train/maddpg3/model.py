#
# coding=utf-8

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from config import logger


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        obs_dim = dim_obs * n_agent
        act_dim = dim_action * n_agent

        # activate func
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()

        self.linear_o_c1 = nn.Linear(obs_dim, 64)
        self.linear_a_c1 = nn.Linear(act_dim, 64)
        self.linear_c2 = nn.Linear(128, 32)
        self.linear_c = nn.Linear(32, 1)

        self.reset_params()

    def reset_params(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        """
        :param obs: (batch, obs_dim)
        :param acts: (batch, acts_dim)
        :return: (batch, 1)
        """
        logger.debug('critic obs shape: {}'.format(obs.shape))
        logger.debug('critic acts: {}'.format(acts.shape))
        x_o = self.LReLU(self.linear_o_c1(obs))
        x_a = self.LReLU(self.linear_a_c1(acts))
        x_cat = th.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_actions):
        super(Actor, self).__init__()
        # activate func
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)

        self.linear_a1 = nn.Linear(dim_obs, 64)
        self.linear_a2 = nn.Linear(64, 32)
        self.linear_a = nn.Linear(32, dim_actions)

        self.reset_params()

    def reset_params(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, obs):
        """
        :param obs: tensor(batch, dim_obs)
        :return: (batch, dim_act)
        """
        x = self.LReLU(self.linear_a1(obs))
        x = self.LReLU(self.linear_a2(x))
        logger.debug("actor out before tanh: {}".format(x))
        policy = self.tanh(self.linear_a(x))

        return policy
