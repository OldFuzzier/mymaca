#
# coding=utf-8

"""
    maddpg: critic: 全部info, 部分img, 全部action
"""

import numpy as np
import torch as th

from train.maddpg.MADDPG import MADDPG
from interface import Environment
from agent.fix_rule.agent import Agent  # todo 更改规则模型
from config import logger
from train.maddpg.agentutil import fighter_rule
from utils import range_transfer

MAX_EPOCH = 5000
MAX_STEP = 200  # default 5000
BATCH_SIZE = 128  # 200
TARGET_REPLACE_ITER = 20  # target update frequency 100
CAPACITY = 5000  # before 1e6, 500
LEARN_INTERVAL = 100  # 学习间隔 100
EPISODES_BEFORE_TRAIN = 10  # 开始训练前的回合数 100
LR = 0.01  # learning rate
GAMMA = 0.9  # reward discount
SCALE_REWARD = 1  # 奖励缩放  # 0.01

RENDER = False
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
AGENT_NAME = 'maddpg'
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 360  # 0-359
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # 0-24 long missile attack + short missile attack + no attack
RADAR_NUM = 11  # 0-10
INTERFERE_NUM = 12  # 0-11
INFO_NUM = 3
JREV_NUM = 40  # todo 如果有detector: 48
ACTION_NUM = 4  # dpg
# ACTION_SCALE = 0.01  # 动作缩放

if __name__ == '__main__':
    # reward_record = []  # 记录每轮训练的奖励
    # a_loss_list = []  # len=10
    # c_loss_list = []  # len=10
    # get agent obs type
    blue_agent = Agent()  # blue agent
    red_agent_obs_ind = AGENT_NAME  # red agent
    blue_agent_obs_ind = blue_agent.get_obs_ind()
    # init model
    maddpg = MADDPG(n_agents=FIGHTER_NUM, dim_info=INFO_NUM, dim_jrev=JREV_NUM, dim_act=ACTION_NUM,
                    batch_size=BATCH_SIZE, capacity=CAPACITY, replace_target_iter=TARGET_REPLACE_ITER,
                    episodes_before_train=EPISODES_BEFORE_TRAIN, learning_rate=LR, gamma=GAMMA,
                    scale_reward=SCALE_REWARD)
    # gpu
    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    # make env
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER, max_step=MAX_STEP)
    # get map info
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # set map info to blue agent
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    red_detector_action = []  # temp

    for i_episode in range(MAX_EPOCH):
        step_cnt = 0
        env.reset()
        total_reward = 0.0  # 每回合所有智能体的总体奖励
        a_loss_list = [0.0 for _ in range(10)]
        c_loss_list = [0.0 for _ in range(10)]
        rr = np.zeros((FIGHTER_NUM,))  # 每回合每个智能体的奖励

        # get obs
        red_obs_dict, blue_obs_dict = env.get_obs()  # output: raw obs结构体

        while True:
            # obs_list = []
            img_list = []  # len == n agents
            info_list = []  # len == n agents
            jrev_list = []  # len == n agents
            action_list = []  # # len == n agents
            red_fighter_action = []  # # len == n agents

            # get blue action
            blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, step_cnt)

            # get red action
            for y in range(red_fighter_num):
                tmp_img_obs = red_obs_dict['fighter'][y]['screen']  # (100, 100, 5)
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)  # (5, 100, 100)
                tmp_info_obs = red_obs_dict['fighter'][y]['info']  # (3, 1)
                tmp_jrev_obs = red_obs_dict['fighter'][y]['j_visible']  # (10, 4)
                tmp_jrev_obs = tmp_jrev_obs.transpose(1, 0)  # (4, 10)

                img_list.append(tmp_img_obs)
                info_list.append(tmp_info_obs)
                jrev_list.append(tmp_jrev_obs)
                # temp
                true_action = np.array([0, 1, 1, 0], dtype=np.int32)
                if not red_obs_dict['fighter'][y]['alive']:
                    # 如果有智能体已经死亡，则默认死亡动作输出
                    action_list.append(np.array([0, 0, 0, 0], dtype=np.int32))
                else:
                    # model policy
                    # tmp_action = maddpg.select_action(y, tmp_img_obs, tmp_info_obs, tmp_jrev_obs)
                    # logger.debug('tmp action: {}'.format(tmp_action))
                    # action_list.append(tmp_action)
                    #
                    # # tmp action transfer to true action
                    # # tmp_action = tmp_action * ACTION_SCALE  # 动作缩放
                    # true_action[0] = range_transfer(tmp_action[0], COURSE_NUM)
                    # true_action[1] = range_transfer(tmp_action[1], RADAR_NUM)
                    # true_action[2] = range_transfer(tmp_action[2], INTERFERE_NUM)
                    # true_action[3] = range_transfer(tmp_action[3], ATTACK_IND_NUM)
                    # logger.info('true action: {}'.format(true_action))

                    # rule policy
                    true_action = fighter_rule(tmp_img_obs, tmp_info_obs, tmp_jrev_obs)
                    action_list.append(true_action)
                red_fighter_action.append(true_action)

            # env step
            red_fighter_action = np.array(red_fighter_action)
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()
            detector_reward = red_detector_reward + red_game_reward
            fighter_reward = red_fighter_reward + red_game_reward
            total_reward += fighter_reward.sum()
            rr += fighter_reward

            # get next obs
            red_obs_dict, blue_obs_dict = env.get_obs()

            # store replay
            # next obs
            next_img_list = []
            next_info_list = []
            next_jrev_list = []
            for y in range(red_fighter_num):
                tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                tmp_info_obs = red_obs_dict['fighter'][y]['info']
                tmp_jrev_obs = red_obs_dict['fighter'][y]['j_visible']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                next_img_list.append(tmp_img_obs)
                next_info_list.append(tmp_info_obs)
                next_jrev_list.append(tmp_jrev_obs)

            # store
            maddpg.memory.push(img_list, info_list, jrev_list, action_list, next_img_list,
                               next_info_list, next_jrev_list, fighter_reward)

            # if done, perform a learn
            if env.get_done():
                if maddpg.episode_done > maddpg.episodes_before_train:
                    logger.info('done and training now begins...')
                    maddpg.update_policy()
                break
            # if not done learn when learn interval
            if maddpg.episode_done > maddpg.episodes_before_train and (step_cnt % LEARN_INTERVAL == 0):
                logger.info('training now begins...')
                c_loss, a_loss = maddpg.update_policy()
                # c_loss_list[y] += c_loss
                # a_loss_list[y] += a_loss

            step_cnt += 1
            logger.info("Episode: {}, step: {}".format(maddpg.episode_done, step_cnt))
        maddpg.episode_done += 1
        logger.info('Episode: %d, reward = %f' % (i_episode, total_reward))

        # 每回合结束存loss
        # loss_w(c_loss_list, a_loss_list)

        # reward_record.append(total_reward)
        # 将每轮奖励写入文件
        # score_w(total_reward)
