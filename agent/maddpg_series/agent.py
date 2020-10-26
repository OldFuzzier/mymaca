#
# coding=utf-8


import os
import copy
import numpy as np

from agent.base_agent import BaseAgent
from agent.maddpg_series.maddpg import MADDPG  # class
from config import logger
from config import MODEL_NAME
from utils import range_transfer
from train.maddpg.agentutil import fighter_rule


DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 360  # 0-359
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # 0-24 long missile attack + short missile attack + no attack
RADAR_NUM = 11  # 0-10
INTERFERE_NUM = 12  # 0-11
INFO_NUM = 3
ACTION_NUM = 4  # dpg


class Agent(BaseAgent):
    def __init__(self):
        """
        Init this agent
        :param size_x: battlefield horizontal size
        :param size_y: battlefield vertical size
        :param detector_num: detector quantity of this side
        :param fighter_num: fighter quantity of this side
        """
        BaseAgent.__init__(self)
        self.obs_ind = 'maddpg'
        # if not os.path.exists('model/maddpg/{}0.pkl'.format(MODEL_NAME)):
        #     logger.info('Error: agent simple model data not exist!')
        #     exit(1)
        # self.maddpg = MADDPG(FIGHTER_NUM, INFO_NUM, ACTION_NUM)

    def set_map_info(self, size_x, size_y, detector_num, fighter_num):
        self.size_x = size_x
        self.size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num

    def __reset(self):
        pass

    def get_action(self, obs_dict, step_cnt):
        """
        get actions
        :param detector_obs_list:
        :param fighter_obs_list:
        :param joint_obs_dict:
        :param step_cnt:
        :return:
        """

        detector_action = []
        fighter_action = []
        for y in range(self.fighter_num):
            tmp_img_obs = obs_dict['fighter'][y]['screen']  # (100, 100, 5)
            tmp_info_obs = obs_dict['fighter'][y]['info']  # (3, )
            tmp_jrev_obs = obs_dict['fighter'][y]['j_visible']  # (10, 4)
            tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)  # (5, 100, 100)
            tmp_jrev_obs = tmp_jrev_obs.transpose(1, 0)  # (4, 10)

            true_action = np.array([0, 1, 1, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                # maddpg policy
                # tmp_action = self.maddpg.select_action(y, tmp_img_obs, tmp_info_obs, tmp_jrev_obs)
                # logger.debug('tmp action: {}'.format(tmp_action))
                # # tmp action transfer to true action
                # true_action[0] = range_transfer(tmp_action[0], COURSE_NUM)
                # true_action[1] = range_transfer(tmp_action[1], RADAR_NUM)
                # true_action[2] = range_transfer(tmp_action[2], INTERFERE_NUM)
                # true_action[3] = range_transfer(tmp_action[3], ATTACK_IND_NUM)
                # logger.info('true action: {}'.format(true_action))

                # rule policy
                true_action = fighter_rule(tmp_img_obs, tmp_info_obs, tmp_jrev_obs)

            fighter_action.append(true_action)
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action
