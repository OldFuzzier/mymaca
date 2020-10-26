#
# coding=utf-8

import torch as th
import torch.nn as nn

from config import logger
from utils import norm


class Critic(nn.Module):
    def __init__(self, n_agent, dim_obs, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_obs = dim_obs
        # self.dim_info = dim_info
        # self.dim_jrev = dim_jrev
        self.dim_action = dim_action
        # info_dim = dim_info * n_agent
        # jrev_dim = dim_jrev * n_agent
        obs_dim = dim_obs * n_agent
        act_dim = dim_action * n_agent

        self.conv1 = nn.Sequential(  # 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.all_obs_fc = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.Tanh(),
        )
        self.all_action_fc = nn.Sequential(
            nn.Linear(512 + act_dim, 512),
            nn.ReLU(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 512
            nn.Linear((25 * 25 * 32 + 512), 256),
            nn.Tanh(),
        )
        self.decision_fc = nn.Linear(256, 1)

    # obs: batch_size * obs_dim
    def forward(self, img, info, jrev, acts):
        """
        :param img: (batch, 10, 5, 10, 10)
        :param info: (batch, 10, 3)
        :param jrev: (batch, 10, 10, 4)
        :param acts: (batch, 4)
        :return: (batch, 1)
        """
        # 归一化
        # img = norm(img)
        # info = norm(info)
        # jrev = norm(jrev)
        # acts = norm(acts)
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        jrev = jrev.view(jrev.size(0), -1)
        obs = th.cat((info, jrev), dim=1)
        obs_feature = self.all_obs_fc(obs)
        obs_acts_combine = th.cat((obs_feature, acts), dim=1)
        obs_feature = self.all_action_fc(obs_acts_combine)
        img_info_combine = th.cat((img_feature.view(img_feature.size(0), -1), obs_feature), dim=1)
        feature = self.feature_fc(img_info_combine)
        return self.decision_fc(feature)


class Actor(nn.Module):
    def __init__(self, dim_obs, dim_actions):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 5  # todo 为什么是100*100*3 ?
            nn.Conv2d(
                in_channels=5,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.obs_fc = nn.Sequential(
            nn.Linear(dim_obs, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, dim_actions)

    def forward(self, img, info, jrev):
        """
        :param img: tensor(?, 5, 100, 100)
        :param info: tensor(?, 3)
        :param jrev: tensor(?, 10, 4)
        :return: (?, 4)
        """
        logger.debug('actor img shape: {}'.format(img.shape))
        logger.debug('actor info shape: {}'.format(info.shape))
        logger.debug('actor jrev shape: {}'.format(jrev.shape))
        # 归一化
        # img = norm(img)
        # info = norm(info)
        # jrev = norm(jrev)
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        jrev = jrev.view(jrev.size(0), -1)  # (?, 40)
        obs = th.cat((info, jrev), dim=1)  # (? 43)
        logger.debug('actor obs: {}'.format(obs.shape))
        obs_feature = self.obs_fc(obs)
        img_info_feature = th.cat((img_feature.view(img_feature.size(0), -1), obs_feature), dim=1)
        logger.debug('actor img_info shape: {}'.format(img_info_feature.shape))
        feature = self.feature_fc(img_info_feature)
        action = self.decision_fc(feature)
        logger.debug('actor action: {}'.format(action))
        return action
