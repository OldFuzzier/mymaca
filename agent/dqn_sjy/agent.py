import os
import random

from agent.base_agent import BaseAgent
from agent.dqn_sjy import dqn
from interface import get_distance
import copy
import numpy as np
import math

from config import logger, MODEL_PATH_AGENT, MODEL_NAME_AGENT
from utils import action2direction

DETECTOR_NUM = 0
FIGHTER_NUM = 10
# FIGHTER1_NUM = 8
# FIGHTER2_NUM = 1
# FIGHTER3_NUM = 1

COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
RADAR_NUM = 11
JAMMING_NUM = 12

ACTION_NUM = 21  # todo : 0 ~ 20
OBS_NUM = 1 + 2 + 2 * 10 + 10 + 10 + 20

STEP_BEFORE_TRAIN = 100

OBS_IND_NAME = 'dqn_sjy'


def getarc360(arc):
    if 0 <= arc <= 180:
        return arc
    elif -180 <= arc < 0:
        return 360 + arc


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
        self.obs_ind = OBS_IND_NAME
        if not os.path.exists('model/{}/{}.pkl'.format(MODEL_PATH_AGENT, MODEL_NAME_AGENT)):
            logger.info('Error: agent maddpg4 model data not exist!')
            exit(1)
        self.fighter_model = dqn.RLFighter(OBS_NUM, ACTION_NUM)
        self.step = 0

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
        # 在观测列表中选第一个目标时为id最小目标（与之前优先选择先观测到的目标不同）
        """
        red_obs_dict_fighter = obs_dict['fighter']
        detector_action = []
        fighter_action = []

        for y in range(self.fighter_num):
            tmp_course = red_obs_dict_fighter[y]['course']  # (1, )

            tmp_pos = red_obs_dict_fighter[y]['pos']  # (2, )
            tmp_l_missile = red_obs_dict_fighter[y]['l_missile']  # (1, )
            tmp_s_missile = red_obs_dict_fighter[y]['s_missile']  # (1, )
            tmp_r_visible_pos = red_obs_dict_fighter[y]['r_visible_pos']  # (10, 2)
            tmp_j_visible_dir = red_obs_dict_fighter[y]['j_visible_dir']  # (10, 1)
            tmp_j_visible_fp = red_obs_dict_fighter[y]['j_visible_fp']  # (10, 1)
            tmp_striking_list = red_obs_dict_fighter[y]['striking_id']  # (10, 1)
            tmp_g_visible_pos = red_obs_dict_fighter[y]['g_visible_pos']  # (10, 2)

            course = tmp_course / 359.  # (1, 1)
            pos = tmp_pos / 1000  # (1, 2)
            r_visible_pos = tmp_r_visible_pos.reshape(1, -1)[0] / 1000  # (1, 20)
            j_visible_dir = tmp_j_visible_dir.reshape(1, -1)[0] / 359.  # (1, 10)
            g_visible_pos = tmp_g_visible_pos.reshape(1, -1)[0] / 1000  #
            striking_list = tmp_striking_list.reshape(1, -1)[0]  # (1, 10)

            tmp_r_visible_pos = tmp_r_visible_pos.transpose(1, 0)  # (2,10)
            tmp_j_visible_dir = tmp_j_visible_dir.transpose(1, 0)  # (1,10)
            tmp_j_visible_fp = tmp_j_visible_fp.transpose(1, 0)  # (1,10)
            # tmp_striking_list = tmp_striking_list.transpose(1, 0)  # (1,10)
            tmp_g_visible_pos = tmp_g_visible_pos.transpose(1, 0)  # (2,10)

            obs = np.concatenate((course, pos, r_visible_pos, j_visible_dir, g_visible_pos, striking_list), axis=0)  #

            # 判断是否战斗单元侦测到可攻击的对象
            farr = np.where(tmp_r_visible_pos[0] >= 0)  # 主动观测列表 farr = [id, id']
            farr1 = np.where(tmp_g_visible_pos[0] >= 0)  # 全局观测列表
            farr2 = np.where(tmp_j_visible_dir[0] >= 0)  # 被动观测列表
            farr_strike = np.where(tmp_striking_list > 0)

            if not red_obs_dict_fighter[y]['alive']:
                true_action = np.array([0, 0, 0, 0], dtype=np.int32)
            else:
                true_action = np.array([0, 1, 1, 0], dtype=np.int32)
                true_action[0] = tmp_course  # 固定航向

                if tmp_l_missile[0] == 0 and tmp_s_missile[0] > 0:
                    # 雷达观测到有敌人
                    if len(farr[0]) > 0:
                        logger.debug('雷达观测到敌人............')
                        print('farr[0]', farr[0])
                        # logger.wait()
                        id_ = random.choice(farr[0])
                        fightpos = [tmp_r_visible_pos[0][id_], tmp_r_visible_pos[1][id_]]
                        print('fightpos', fightpos)  # ........ceshi
                        action_id = id_ + 1  # id为索引号+1
                        true_action[0] = getarc360(
                            int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
                        # true_action[2] = 1
                        # if get_distance(fightpos[0], fightpos[1], tmp_pos[0], tmp_pos[1]) > 40:
                        # if step_cnt % 5 == 0:
                        if get_distance(fightpos[0], fightpos[1], tmp_pos[0], tmp_pos[1]) <= 50:  # todo 加攻击距离
                            true_action[3] = action_id + 10
                        # -------------------------------
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        # logger.wait()
                        # step = step_cnt
                        # step += 1
                    # 战机被动观测列表观测到敌人
                    elif len(farr2[0]) > 0:
                        logger.debug('len(farr2[0]: {}'.format(len(farr2[0])))
                        logger.debug('被动观测列表敌人频点............')
                        print(y + 1, 'sj_farr2[0]', farr2[0])
                        id_ = random.choice(farr2[0])
                        print(y + 1, 'sj_id_', id_)
                        action_id = id_ + 1
                        # print('true_action[0]', true_action[0])
                        # logger.wait()
                        print(y + 1, 'true_action[2]', true_action[2])
                        # logger.wait()
                        # if step % 5 == 0:
                        # if id_ in farr[0]:
                        true_action[0] = tmp_j_visible_dir[0][id_]  # todo

                        # true_action[3] = action_id + 10  # todo
                        # if len(farr2[0]) > 1:
                        #     true_action[2] = 11
                        # else:
                        true_action[2] = tmp_j_visible_fp[0][id_]
                        # -------------------------------
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        # logger.wait()
                        # step += 1
                    # joint全局列表被动观测到有敌人
                    elif len(farr1[0]) > 0:
                        logger.debug('joint列表被动观测到有敌人............')
                        id_ = random.choice(farr1[0])
                        fightpos = [tmp_g_visible_pos[0][id_], tmp_g_visible_pos[1][id_]]
                        action_id = id_ + 1
                        true_action[0] = getarc360(
                            int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
                        if get_distance(fightpos[0], fightpos[1], tmp_pos[0], tmp_pos[1]) <= 50:  # todo 加攻击距离
                            true_action[3] = action_id + 10
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        # logger.wait()
                    else:
                        logger.debug('没有探测到敌人............')
                        # tmp_course[0] = int(tmp_course[0])
                        # if tmp_pos[0] == 0:
                        #     if tmp_pos[1] == 0 or tmp_pos[1] == 1000:
                        #         true_action[0] = getarc360(int(180 - tmp_course[0]))
                        #     else:
                        #         true_action[0] = tmp_course[0] - 250 if tmp_course > 270 else tmp_course[0] - 110  # todo
                        # elif tmp_pos[0] == 1000:
                        #     if tmp_pos[1] == 0 or tmp_pos[1] == 1000:
                        #         true_action[0] = getarc360(int(180 - tmp_course[0]))
                        #     else:
                        #         true_action[0] = getarc360(int(tmp_course[0] - 110)) if tmp_course < 90 else getarc360(int(tmp_course[0] - 250))  # todo
                        # elif tmp_pos[1] == 0 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                        #     true_action[0] = getarc360(int(tmp_course[0] - 110)) if tmp_course < 180 else getarc360(int(tmp_course[0] - 250))  # todo
                        # elif tmp_pos[1] == 1000 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                        #     true_action[0] = getarc360(int(tmp_course[0] - 250)) if tmp_course < 180 else getarc360(int(tmp_course[0] - 110))
                        logger.debug('.............没有探测到敌人............')
                        if tmp_pos[0] == 1000:
                            true_action[0] = 180
                        elif tmp_pos[0] == 0:
                            true_action[0] = 0
                        else:
                            true_action[0] = tmp_course[0]
                        if tmp_pos[1] == 0:
                            true_action[0] = 90
                        if tmp_pos[1] == 1000:
                            true_action[0] = 270


                elif tmp_l_missile[0] > 0:
                    if len(farr[0]) > 0:
                        id_ = random.choice(farr[0])
                        fightpos = [tmp_r_visible_pos[0][id_], tmp_r_visible_pos[1][id_]]
                        # true_action[0] = getarc360(
                        #     int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
                        if get_distance(fightpos[0], fightpos[1], tmp_pos[0], tmp_pos[1]) <= 120:  # todo 加攻击距离
                            true_action[3] = id_ + 1
                        logger.debug('雷达发现敌人长导弹打击: {}'.format(true_action[3]))
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        # logger.wait()
                    elif len(farr2[0]) > 0:
                        # print('...............farr2[0]', farr2[0])
                        # logger.wait()
                        id_ = random.choice(farr2[0])
                        # logger.wait()
                        # if id_ in farr[0]:
                        true_action[0] = tmp_j_visible_dir[0][id_]  # todo
                        # true_action[3] = id_ + 1           # todo
                        print('true_action[3]', true_action[3])
                        # if len(farr2[0]) > 1:   # todo
                        #     true_action[2] = 11
                        # else:
                        true_action[2] = tmp_j_visible_fp[0][id_]
                        print(y + 1, '-------true_action[2]', true_action[2])
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        # logger.wait()

                        # logger.wait()
                        logger.debug('被动观测发现敌人长导弹打击: {}'.format(true_action[3]))

                    elif len(farr1[0]) > 0:
                        id_ = random.choice(farr1[0])
                        fightpos = [tmp_g_visible_pos[0][id_], tmp_g_visible_pos[1][id_]]
                        # true_action[0] = getarc360(
                        #     int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
                        if get_distance(fightpos[0], fightpos[1], tmp_pos[0], tmp_pos[1]) <= 120:  # todo 加攻击距离
                            true_action[3] = id_ + 1
                        logger.debug('全局被动发现敌人长导弹打击: {}'.format(true_action[3]))
                        # if len(farr_strike[0]) > 0:
                        #     true_action[0] = true_action[0] + 6
                        #     logger.wait()
                    else:
                        logger.debug('.............没有探测到敌人............')
                        if tmp_pos[0] == 1000:
                            true_action[0] = 180
                        elif tmp_pos[0] == 0:
                            true_action[0] = 0
                        else:
                            true_action[0] = tmp_course[0]
                        if tmp_pos[1] == 0:
                            true_action[0] = 90
                        if tmp_pos[1] == 1000:
                            true_action[0] = 270
                        # tmp_course[0] = int(tmp_course[0])
                        # if tmp_pos[0] == 0:
                        #     if tmp_pos[1] == 0 or tmp_pos[1] == 1000:
                        #         true_action[0] = getarc360(int(180 - tmp_course[0]))
                        #     else:
                        #         true_action[0] = tmp_course[0] - 250 if tmp_course > 270 else tmp_course[
                        #                                                                              0] - 110  # todo
                        # elif tmp_pos[0] == 1000:
                        #     if tmp_pos[1] == 0 or tmp_pos[1] == 1000:
                        #         true_action[0] = getarc360(int(180 - tmp_course[0]))
                        #     else:
                        #         true_action[0] = getarc360(int(tmp_course[0] - 110)) if tmp_course < 90 else getarc360(
                        #             int(tmp_course[0] - 250))  # todo
                        # elif tmp_pos[1] == 0 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                        #     true_action[0] = getarc360(int(tmp_course[0] - 110)) if tmp_course < 180 else getarc360(int(tmp_course[0] - 250))  # todo
                        # elif tmp_pos[1] == 1000 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                        #     true_action[0] = getarc360(int(tmp_course[0] - 250)) if tmp_course < 180 else getarc360(int(tmp_course[0] - 110))

                elif tmp_l_missile[0] == 0 and tmp_s_missile[0] == 0:
                    true_action[1] = 0
                    if len(farr1[0]) > 0:
                        id_ = random.choice(farr1[0])
                        fightpos = [tmp_g_visible_pos[0][id_], tmp_g_visible_pos[1][id_]]
                        e_course = getarc360(
                            int(math.atan2(fightpos[1] - tmp_pos[1], fightpos[0] - tmp_pos[0]) * 180 / math.pi))
                        true_action[0] = getarc360(180 - e_course)
                    elif len(farr2[0]) > 0:
                        id_ = random.choice(farr2[0])
                        true_action[0] = getarc360(180 - tmp_j_visible_dir[0][id_])
                        true_action[2] = tmp_j_visible_fp[0][id_]
                    else:
                        if tmp_pos[0] == 0:
                            if tmp_pos[1] == 0:
                                true_action[0] = 90 if 45 < tmp_course < 225 else 0
                            elif tmp_pos[1] == 1000:
                                true_action[0] = 270 if 135 < tmp_course < 315 else 0
                            else:
                                true_action[0] = 270 if tmp_course > 180 else 90
                        elif tmp_pos[0] == 1000:
                            if tmp_pos[1] == 0:
                                true_action[0] = 180 if 135 < tmp_course < 315 else 90
                            elif tmp_pos[1] == 1000:
                                true_action[0] = 180 if 45 < tmp_course < 225 else 270
                            else:
                                true_action[0] = 270 if tmp_course > 180 else 90
                        elif tmp_pos[1] == 0 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                            true_action[0] = 0 if tmp_course > 270 else 180
                        elif tmp_pos[1] == 1000 and tmp_pos[0] != 0 and tmp_pos[0] != 1000:
                            true_action[0] = 0 if tmp_course < 90 else 180

                if step_cnt == 1:
                    arc_center = getarc360(
                        int(math.atan2(500 - tmp_pos[1], 500 - tmp_pos[0]) * 180 / math.pi))
                    if 90 < arc_center < 270:
                        true_action[0] = 180
                    else:
                        true_action[0] = 0

            # 添加动作action[0]
            if step_cnt > STEP_BEFORE_TRAIN and (len(farr[0]) <= 0 and len(farr1[0]) <= 0 and len(farr2[0]) <= 0):
                tmp_action = self.fighter_model.choose_action(obs)
                # if len(farr_strike[0]) <= 0:
                # if len(farr2[0]) > 0:
                true_action[0] = action2direction(true_action[0], tmp_action, ACTION_NUM)

            # print(y+1, '偏角', tmp_action)

            fighter_action.append(copy.deepcopy(true_action))
        # print("fighter_action", fighter_action)
        # logger.debug()
        fighter_action = np.array(fighter_action)
        # print("fighter_action", fighter_action)
        # logger.debug()
        return detector_action, fighter_action
