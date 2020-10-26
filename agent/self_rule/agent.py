#
# coding=utf-8


import numpy as np

from agent.base_agent import BaseAgent
from config import logger
from rule.agentutil_stable import fighter_rule


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
        self.obs_ind = 'self_rule'

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
            tmp_l_missile = obs_dict['fighter'][y]['l_missile']  # (1, )
            tmp_s_missile = obs_dict['fighter'][y]['s_missile']  # (1, )
            tmp_r_visible_pos = obs_dict['fighter'][y]['r_visible_pos']  # (10, 2)
            tmp_j_visible_fp = obs_dict['fighter'][y]['j_visible_fp']  # rule use
            tmp_j_visible_dir = obs_dict['fighter'][y]['j_visible_dir']  # (10, 1)
            tmp_g_striking_pos = obs_dict['fighter'][y]['g_striking_pos']  # (10, 2)
            tmp_r_visible_dis = obs_dict['fighter'][y]['r_visible_dis']  # (10, 1)
            tmp_striking_id = obs_dict['fighter'][y]['striking_id']
            true_action = np.array([0, 1, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                # rule policy
                # true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                #                            tmp_r_visible_dis, tmp_j_visible_dir, tmp_j_visible_fp,
                #                            tmp_g_striking_pos, step_cnt)
                true_action = fighter_rule(tmp_course, tmp_pos, tmp_l_missile, tmp_s_missile, tmp_r_visible_pos,
                                           tmp_r_visible_dis, tmp_j_visible_dir, tmp_j_visible_fp,
                                           tmp_striking_id, tmp_g_striking_pos, step_cnt)
                logger.debug('true action rule out: {}'.format(true_action))
            logger.debug('true action: {}'.format(true_action))
            fighter_action.append(true_action)
        fighter_action = np.array(fighter_action)

        return detector_action, fighter_action
