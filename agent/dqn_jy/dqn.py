#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Gao Fang
@contact: gaofang@cetc.com.cn
@software: PyCharm
@file: dqn.py
@time: 2018/7/25 0025 13:38
@desc:
"""
import os
import torch
import torch.nn as nn
import numpy as np


class NetFighter(nn.Module):
    def __init__(self, n_actions):
        super(NetFighter, self).__init__()
        self.conv1 = nn.Sequential(     # 100 * 100 * 3
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
        self.conv2 = nn.Sequential(     # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(    # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action

# Deep Q Network off-policy
class RLFighter:
    def __init__(
            self,
            n_actions,
    ):
        self.n_actions = n_actions
        self.gpu_enable = torch.cuda.is_available()

        self.target_net = NetFighter(self.n_actions)
        if self.gpu_enable:
            print('GPU Available!!')
            self.target_net = self.target_net.cuda()
            #self.target_net.load_state_dict(torch.load('model/simple/model.pkl'))
            self.target_net.load_state_dict(torch.load('model/dqn_jy/model_000015200.pkl'))
        else:
            self.target_net.load_state_dict(torch.load('model/dqn_jy/model_000015200.pkl', map_location=lambda storage, loc: storage))

    def choose_action(self, img_obs, info_obs):
        #torch.squeeze() 这个函数对数据的维度进行压缩，去掉维数为1的的维度，比如是一行或者一列这种
        #torch.unsqueeze() 这个函数对数据维度进行扩充。给指定位置加上维数为一的维度
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        actions_value = self.target_net(img_obs, info_obs)
        #按维度dim 返回最大值torch.max(a,0) 返回每一列中最大值的那个元素，且返回索引（返回最大元素在这一列的行索引）
        action = torch.max(actions_value, 1)[1]
        if self.gpu_enable:
            action = action.cpu()
        action = action.numpy()
        return action

class NetDetector(nn.Module):
    def __init__(self, n_actions):
        super(NetDetector, self).__init__()
        self.conv1 = nn.Sequential(     # 100 * 100 * 3
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(     # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),            # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(    # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action

