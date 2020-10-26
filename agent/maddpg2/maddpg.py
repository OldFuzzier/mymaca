#
# coding=utf-8

import torch as th
import torch.nn as nn
from config import logger
from config import GPU_CONFIG
from config import MODEL_NAME


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
        logger.debug("actor out before tanh: {}".format(self.linear_a(x)))
        policy = self.tanh(self.linear_a(x))
        return policy


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]

        self.n_agents = n_agents
        self.n_obs = dim_obs
        self.n_actions = dim_act
        self.use_cuda = GPU_CONFIG.use_cuda

        if self.use_cuda:
            logger.info("GPU Available!!")
        for i, x in enumerate(self.actors):
            if self.use_cuda:
                x.to(GPU_CONFIG.device)
                x.load_state_dict(th.load("model/maddpg2/{}{}.pkl".format(MODEL_NAME, i)))
            else:
                x.load_state_dict(th.load("model/maddpg2/{}{}.pkl".format(MODEL_NAME, i),
                                          map_location=lambda storage, loc: storage))

    def select_action(self, agent_i, obs):
        """
        :param agent_i: int
        :param img_obs: ndarray
        :param info_obs: ndarray
        :return: action: ndarray
        """
        # ndarray to tensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        obs = th.unsqueeze(FloatTensor(obs), 0)
        # action
        action = self.actors[agent_i](obs).squeeze()
        logger.info('select action: {}'.format(action))
        action = th.clamp(action, -1., 1.)
        logger.debug('select action+clamp: {}'.format(action))

        # tensor to ndarray
        if self.use_cuda:
            action = action.data.cpu()
        else:
            action = action.detach()
        action = action.numpy()

        return action
