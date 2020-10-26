#
# coding=utf-8


import os
import numpy as np

from agent.base_agent import BaseAgent
from config import logger, IS_DISPERSED, MODEL_PATH, MODEL_NAME
from utils import range_transfer
from rule.agentutil4 import fighter_rule

if IS_DISPERSED:
    from agent.maddpg2.maddpg_dispersed import MADDPG
else:
    from agent.maddpg2.maddpg import MADDPG  # class

DETECTOR_NUM = 0
FIGHTER_NUM = 10
OBS_NUM = 2 + 1 + 2 * 10 + 10 + 2 * 10
ACTION_NUM = 12 if IS_DISPERSED else 1  # # dispersed num, ddpg


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
        self.obs_ind = 'maddpg2'
        if not os.path.exists('model/{}/{}0.pkl'.format(MODEL_PATH, MODEL_NAME)):
            logger.info('Error: agent maddpg2 model data not exist!')
            exit(1)
        self.maddpg = MADDPG(FIGHTER_NUM, OBS_NUM, ACTION_NUM)

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
            tmp_course = obs_dict['fighter'][y]['course']  # (1, )
            tmp_pos = obs_dict['fighter'][y]['pos']  # (2, )
            tmp_r_visible_pos = obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
            tmp_j_visible_fp = obs_dict['fighter'][y]['j_visible_fp']  # (10, 1)
            tmp_l_missile = obs_dict['fighter'][y]['l_missile']  # rule use
            tmp_s_missile = obs_dict['fighter'][y]['s_missile']  # rule use
            tmp_j_visible_fp = obs_dict['fighter'][y]['j_visible_fp']  # rule use
            tmp_j_visible_dir = obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)  # rule use
            tmp_g_visible_pos = obs_dict['fighter'][y]['g_visible_pos']  # (10, 2)
            # model obs change, 归一化
            course = tmp_course / 359.
            pos = tmp_pos / self.size_x
            r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / self.size_x  # (20,)
            j_visible_fp = tmp_j_visible_fp.reshape(1, -1)[0] / 359.  # (10,)
            g_visible_pos = tmp_g_visible_pos.reshape(1, -1)[0] / self.size_x  # (20,)
            obs = np.concatenate((course, pos, r_visible_pos, j_visible_fp, g_visible_pos), axis=0)
            logger.debug('obs: {}'.format(obs))

            true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                # model policy
                tmp_action_i = self.maddpg.select_action(y, obs)
                logger.debug('tmp action i: {}'.format(tmp_action_i))
                # rule policy
                true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                                           tmp_j_visible_dir, tmp_j_visible_fp, tmp_g_visible_pos, step_cnt)
                logger.debug('true aciton rule out: {}'.format(true_action))
                # 添加动作 todo
                true_action[2] = np.argmax(tmp_action_i) if IS_DISPERSED else range_transfer(tmp_action_i, 11)
                if true_action[2] == 11:
                    logger.info('agent {}: right'.format(y + 1))
                    logger.wait()
                logger.info('true action: {}'.format(true_action))

            fighter_action.append(true_action)
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action
