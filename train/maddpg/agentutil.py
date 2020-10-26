#
# coding=utf-8

"""
    为训练添加规则
"""

import math
import numpy as np

from config import logger


def getarc360(arc):
    """
    航向转换
    :param arc: int, 航向
    :return:
    """
    if 0 <= arc <= 180:
        return arc
    elif -180 <= arc < 0:
        return 360 + arc


def fighter_rule_tmp(red_fighter_num, red_obs_dict):
    """
    飞机打击规则
    1 根据雷达观测>被动观测>全局观测获取地方单位，获取到敌方单位id或者频点
    2 根据远、近导弹剩余量判断使用什么武器打击
    3 进行打击或者干扰
    4 如果没有发现敌人，根据自身位置航行
    5 如果没有导弹，将雷达频点与干扰频点关闭
    :param red_fighter_num: int, agent数量
    :param red_obs_dict: dict
    :return:  true_action_list: list, 战斗机行动
    """
    true_action_list = []
    red_obs_dict_fighter = red_obs_dict['fighter']  # 只抽取战斗机状态

    for y in range(red_fighter_num):
        tmp_img_obs = red_obs_dict_fighter[y]['screen']
        tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
        tmp_info_obs = red_obs_dict_fighter[y]['info']
        tmp_jrev_obs = red_obs_dict_fighter[y]['j_visible']

        # 判断是否战斗单元侦测到可攻击的对象   tmp_img_obs[0][r][c]=id tmp_img_obs[1][r][c]=type
        farr = np.where(tmp_img_obs[0] > 0)
        farr1 = np.where(tmp_img_obs[3] > 0)
        farr2 = np.where(tmp_jrev_obs[0] > 0)

        if not red_obs_dict_fighter[y]['alive']:
            true_action_list.append(np.array([0, 0, 0, 0], dtype=np.int32))
        else:
            # 判断位置值为255 的自己位置 oarr[0][4] oarr[1][4]
            oarr = np.where(tmp_img_obs[2] == 255)
            curpos = [oarr[0][int((len(oarr[0]) - 1) / 2)], oarr[1][int((len(oarr[1]) - 1) / 2)]]
            true_action = np.array([0, 1, 1, 0], dtype=np.int32)
            fightpos = [0, 0]  # 敌方位置
            
            if tmp_info_obs[1] == 0 and tmp_info_obs[2] > 1:
                # 雷达观测到有敌人
                if len(farr[0]) > 0:
                    logger.debug('雷达观测到敌人............')
                    fightpos[0] = farr[0][int((len(farr[0]) - 1) / 2)]
                    fightpos[1] = farr[1][int((len(farr[1]) - 1) / 2)]
                    # 敌人id【】
                    action_id = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
                    logger.debug('this_action_id: {}'.format(action_id))
                    true_action[0] = getarc360(
                        int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1]) * 180 / math.pi))
                    true_action[2] = 11
                    true_action[3] = action_id[0] + 10
                # joint全局列表被动观测到有敌人
                if len(farr1[0]) > 0:
                    logger.debug('joint列表被动观测到有敌人............')
                    fightpos = [farr1[0][int((len(farr[0]) - 1) / 2)], farr1[1][int((len(farr[1]) - 1) / 2)]]
                    # 敌人id【】
                    action_id = [tmp_img_obs[3][farr1[0][0]][farr1[1][0]]]
                    logger.debug('this_tmp_action1: {}'.format(action_id))
                    true_action[0] = getarc360(
                        int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1]) * 180 / math.pi))
                    true_action[2] = 11
                    true_action[3] = action_id[0] + 10

                # 战机被动观测列表观测到敌人
                if len(farr2[0]) > 0:
                    logger.debug('被动观测列表敌人频点............')
                    true_action[2] = tmp_jrev_obs[3]
                    true_action[1] = true_action[1] + 1
                    true_action[0] = tmp_jrev_obs[2] + tmp_info_obs[0]
                    true_action[3] = tmp_jrev_obs[0] + 10

                else:
                    logger.debug('没有探测到敌人............')
                    if curpos[1] >= 90:
                        true_action[0] = 180
                        true_action[1] = 0
                    elif curpos[1] <= 10:
                        true_action[0] = 0
                        true_action[1] = true_action[1] + 1
                    else:
                        true_action[0] = tmp_info_obs[0]
                    if curpos[0] == 0:
                        true_action[0] = 120
                    elif curpos[0] == 98:
                        true_action[0] = 330

            if tmp_info_obs[1] > 0:
                if len(farr[0]) > 0:
                    action_id = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
                    logger.debug('雷达发现敌人长导弹打击: {}'.format(action_id))
                    true_action[3] = action_id[0]
                elif len(farr2[0]) > 0:
                    action_id = [tmp_jrev_obs[0]]
                    logger.debug('被动观测发现敌人长导弹打击: {}'.format(action_id))
                    true_action[2] = tmp_jrev_obs[3]
                    true_action[3] = action_id[0]

                elif len(farr1[0]) > 0:
                    action_id = [tmp_img_obs[3][farr1[0][0]][farr1[1][0]]]
                    logger.debug('全局被动发现敌人长导弹打击: {}'.format(action_id))
                    true_action[3] = action_id[0]
            elif tmp_info_obs[2] == 0 and tmp_info_obs[2] == 0:
                logger.debug('没有导弹............')
                # config.logger.debug()
                true_action[1] = 0
                true_action[2] = 0

            true_action_list.append(true_action)
    return true_action_list


def fighter_rule(tmp_img_obs, tmp_info_obs, tmp_jrev_obs):
    """
    飞机打击规则
    1 根据雷达观测>被动观测>全局观测获取地方单位，获取到敌方单位id或者频点
    2 根据远、近导弹剩余量判断使用什么武器打击
    3 进行打击或者干扰
    4 如果没有发现敌人，根据自身位置航行
    5 如果没有导弹，将雷达频点与干扰频点关闭
    :param tmp_img_obs: dict
    :param tmp_info_obs: dict
    :param tmp_jrev_obs: dict
    :return:  true_action: array, 战斗机行动
    """

    # 判断是否战斗单元侦测到可攻击的对象   tmp_img_obs[0][r][c]=id tmp_img_obs[1][r][c]=type
    farr = np.where(tmp_img_obs[0] > 0)  # 主动观测列表  # todo 是一个飞机一个点么
    farr1 = np.where(tmp_img_obs[3] > 0)  # 全局观测列表
    farr2 = np.where(tmp_jrev_obs[0] > 0)  # 被动观测列表

    true_action = np.array([0, 1, 1, 0], dtype=np.int32)

    # 判断位置值为255 的自己位置 oarr[0][4] oarr[1][4]
    oarr = np.where(tmp_img_obs[2] == 255)
    curpos = [oarr[0][int((len(oarr[0]) - 1) / 2)], oarr[1][int((len(oarr[1]) - 1) / 2)]]

    if tmp_info_obs[1] == 0 and tmp_info_obs[2] > 1:
        # 雷达观测到有敌人
        if len(farr[0]) > 0:
            logger.debug('雷达观测到敌人............')
            fightpos = [farr[0][int((len(farr[0]) - 1) / 2)], farr[1][int((len(farr[1]) - 1) / 2)]]
            action_id = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
            true_action[0] = getarc360(
                int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1]) * 180 / math.pi))
            true_action[2] = 11
            true_action[3] = action_id[0] + 10
        # joint全局列表被动观测到有敌人
        if len(farr1[0]) > 0:
            logger.debug('joint列表被动观测到有敌人............')
            fightpos = [farr1[0][int((len(farr[0]) - 1) / 2)], farr1[1][int((len(farr[1]) - 1) / 2)]]
            action_id = [tmp_img_obs[3][farr1[0][0]][farr1[1][0]]]
            true_action[0] = getarc360(
                int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1]) * 180 / math.pi))
            true_action[2] = 11
            true_action[3] = action_id[0] + 10
        # 战机被动观测列表观测到敌人
        if len(farr2[0]) > 0:
            logger.debug('len(farr2[0]: {}'.format(len(farr2[0])))
            logger.debug('被动观测列表敌人频点............')
            true_action[2] = tmp_jrev_obs[3][farr2[0][0]]  #
            true_action[1] = true_action[1] + 1
            true_action[0] = tmp_jrev_obs[2][farr2[0][0]] + tmp_info_obs[0]  #
            true_action[3] = tmp_jrev_obs[0][farr2[0][0]] + 10  #
        else:
            logger.debug('没有探测到敌人............')
            if curpos[1] >= 90:
                true_action[0] = 180
                true_action[1] = 0
            elif curpos[1] <= 10:
                true_action[0] = 0
                true_action[1] = true_action[1] + 1
            else:
                true_action[0] = tmp_info_obs[0]
            if curpos[0] == 0:
                true_action[0] = 120
            elif curpos[0] == 98:
                true_action[0] = 330

    if tmp_info_obs[1] > 0:
        if len(farr[0]) > 0:
            action_id = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
            logger.debug('雷达发现敌人长导弹打击: {}'.format(action_id))
            true_action[3] = action_id[0]
        elif len(farr2[0]) > 0:
            # action_id = [tmp_jrev_obs[0]]
            logger.debug('被动观测发现敌人长导弹打击: {}'.format(farr2))
            true_action[2] = tmp_jrev_obs[3][farr2[0][0]]  #
            true_action[3] = tmp_jrev_obs[0][farr2[0][0]]  #
            logger.debug('干扰频点: {}'.format(true_action[2]))

        elif len(farr1[0]) > 0:
            action_id = [tmp_img_obs[3][farr1[0][0]][farr1[1][0]]]
            logger.debug('全局被动发现敌人长导弹打击: {}'.format(action_id))
            true_action[3] = action_id[0]
    elif tmp_info_obs[2] == 0 and tmp_info_obs[2] == 0:
        logger.debug('没有导弹............')
        true_action[1] = 0
        true_action[2] = 0

    return true_action