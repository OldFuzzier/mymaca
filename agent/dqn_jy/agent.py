import os
from agent.base_agent import BaseAgent
from agent.dqn_jy import dqn
import copy
import numpy as np
import math
import random
from rule import agentutil_jy_dqn as agentutil

DETECTOR_NUM = 0
FIGHTER_NUM = 10
#COURSE_NUM = 16
COURSE_NUM = 2
#ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1 # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM


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
        # self.preposs = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
        #            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.obs_ind = 'dqn_jy'
        if not os.path.exists('model/dqn_jy/model_000015200.pkl'):
            print('class Agent(BaseAgent) Error: agent test1 model data not exist!')
            exit(1)
        self.fighter_model = dqn.RLFighter(ACTION_NUM)

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
            true_action = np.array([0, 0, 0, 0], dtype=np.int32)
            if obs_dict['fighter'][y]['alive']:
                true_action = np.array([0, 1, 1, 0], dtype=np.int32)
                tmp_img_obs = obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = obs_dict['fighter'][y]['info']
                tmp_action = self.fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
                # 判断是否战斗单元侦测到可攻击的对象   tmp_img_obs[0][r][c]=id tmp_img_obs[1][r][c]=type
                farr = np.where(tmp_img_obs[0] > 0)
                oarr = np.where(tmp_img_obs[2] == 255)
                curpos = [oarr[0][int((len(oarr[0]) - 1) / 2)], oarr[1][int((len(oarr[1]) - 1) / 2)]]
                if tmp_info_obs[1]==0 and tmp_info_obs[2]>1:
                    if len(farr[0]) > 0:
                        fightpos = [farr[0][int((len(farr[0]) - 1) / 2)], farr[1][int((len(farr[1]) - 1) / 2)]]
                        tmp_action = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
                        true_action[0] = agentutil.getarc360(
                            int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1]) * 180 / math.pi))
                    else:
                        #true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                        if curpos[1] >= 90:
                            true_action[0] = 180
                        elif curpos[1] <= 10:
                            true_action[0] = 0
                        else:
                            true_action[0] = tmp_info_obs[0]

                        if curpos[0] == 0:
                            true_action[0] = 120
                        elif curpos[0] == 98:
                            true_action[0] = 330
                else:
                    true_action[0] = 0
                if tmp_info_obs[1] > 0:
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                elif tmp_info_obs[2] > 0:
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM) + 10
                else:
                    true_action[3] = 0
                # print("true_action", true_action)
            fighter_action.append(copy.deepcopy(true_action))
        # print("fighter_action",fighter_action)
        fighter_action = np.array(fighter_action)
        return detector_action, fighter_action
