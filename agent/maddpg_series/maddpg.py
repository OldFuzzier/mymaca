#
# coding=utf-8

import torch as th
import torch.nn as nn
from config import logger
from config import GPU_CONFIG
from config import MODEL_NAME


class Actor(nn.Module):
    def __init__(self, dim_info, dim_actions):
        super(Actor, self).__init__()
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
        self.info_fc = nn.Sequential(
            nn.Linear(dim_info, 256),
            nn.ReLU(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, dim_actions)

    def forward(self, img, info):
        """
        :param img: tensor(?, 5, 100, 100)
        :param info: tensor(?, 3)
        :return:
        """
        logger.debug('actor img shape: {}'.format(img.shape))
        logger.debug('actor info shape: {}'.format(info.shape))
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        img_info_feature = th.cat(
            (img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
            dim=1)
        logger.debug('actor img_info shape: {}'.format(img_info_feature.shape))
        feature = self.feature_fc(img_info_feature)
        action = self.decision_fc(feature)
        logger.debug('actor action: {}'.format(action))
        return action


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.use_cuda = GPU_CONFIG.use_cuda

        if self.use_cuda:
            logger.info("GPU Available!!")
        for i, x in enumerate(self.actors):
            if self.use_cuda:
                x.to(GPU_CONFIG.device)
                x.load_state_dict(th.load("model/maddpg/{}{}.pkl".format(MODEL_NAME, i)))
            else:
                x.load_state_dict(th.load("model/maddpg/{}{}.pkl".format(MODEL_NAME, i),
                                          map_location=lambda storage, loc: storage))

    def select_action(self, agent_i, img_obs, info_obs):
        """
        :param agent_i: int
        :param img_obs: ndarray
        :param info_obs: ndarray
        :return: action: ndarray
        """
        # ndarray to tensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        img_obs = th.unsqueeze(FloatTensor(img_obs), 0)
        info_obs = th.unsqueeze(FloatTensor(info_obs), 0)
        # action
        action = self.actors[agent_i](img_obs, info_obs).squeeze()
        logger.debug('select action: {}'.format(action))
        # 加噪声
        action = th.tanh(action)
        logger.debug('select action+tanh: {}'.format(action))

        # tensor to ndarray
        if self.use_cuda:
            action = action.data.cpu()  # todo 搜索tensor.data.cpu()的用法
        else:
            action = action.detach()
        action = action.numpy()

        return action
