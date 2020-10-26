#
# coding=utf-8

from torch.optim import Adam
import torch.nn as nn
import numpy as np

from train.maddpg.model import Critic, Actor
import torch as th
from copy import deepcopy
from train.maddpg.memory2 import ReplayMemory, Experience
from config import GPU_CONFIG
from config import logger

SCORE_FILE_NAME = 'train/maddpg/pics/score.txt'
LOSS_FILE_NAME = 'train/maddpg/pics/loss.txt'


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_info, dim_jrev, dim_act, batch_size, capacity, replace_target_iter,
                 episodes_before_train, learning_rate, gamma, scale_reward):
        self.actors = [Actor(dim_info + dim_jrev, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_info + dim_jrev, dim_act) for i in range(n_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_info = dim_info  # info
        self.n_jrev = dim_jrev  # jrev
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = GPU_CONFIG.use_cuda

        self.GAMMA = gamma
        self.scale_reward = scale_reward
        self.tau = 0.01  # 替换网络比例
        self.replace_target_iter = replace_target_iter
        self.episodes_before_train = episodes_before_train
        self.learn_step_counter = 0
        self.episode_done = 0

        self.var = [0.1 for _ in range(n_agents)]  # todo 1
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=learning_rate) for x in self.critics]  # lr: 0.00.1
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=learning_rate) for x in self.actors]  # lr: 0.0001

        if self.use_cuda:
            logger.info("GPU Available!!")
            for x in self.actors:
                x.to(GPU_CONFIG.device)
            for x in self.critics:
                x.to(GPU_CONFIG.device)
            for x in self.actors_target:
                x.to(GPU_CONFIG.device)
            for x in self.critics_target:
                x.to(GPU_CONFIG.device)

    def update_policy(self):
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size)
            batch = Experience(*zip(*transitions))  # class(list)
            # state_batch: batch_size x n_agents x dim_obs
            img_batch = FloatTensor(np.array(batch.img_states))
            logger.debug("learn img batch: {}".format(img_batch.shape))  # torch.Size([batch, 10, 5, 100, 100])
            logger.debug("learn a agent img batch: {}".format(img_batch[:, 0, :].shape))  # torch.Size([batch, 5, 100, 100])
            info_batch = FloatTensor(np.array(batch.info_states))
            logger.debug('learn info batch: {}'.format(info_batch.shape))
            jrev_batch = FloatTensor(np.array(batch.jrev_states))
            logger.debug('learn info jrev_batch: {}'.format(jrev_batch.shape))  # torch.Size([batch, 10, 10, 4])
            action_batch = FloatTensor(np.array(batch.actions))
            logger.debug('learn action batch: {}'.format(action_batch.shape))  # torch.Size([batch, 10, 4])
            reward_batch = FloatTensor(np.array(batch.rewards))
            logger.debug('learn reward batch: {}'.format(reward_batch.shape))
            next_img_batch = FloatTensor(np.array(batch.next_img_states))
            logger.debug('learn next img batch: {}'.format(next_img_batch.shape))
            next_info_batch = FloatTensor(np.array(batch.next_info_states))
            logger.debug('learn next info batch: {}'.format(next_info_batch.shape))
            next_jrev_batch = FloatTensor(np.array(batch.next_jrev_states))
            logger.debug('learn next jrev batch: {}'.format(next_jrev_batch.shape))

            # for current agent
            whole_info = info_batch.view(self.batch_size, -1)
            whole_jrev = jrev_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](img_batch[:, agent, :], whole_info, whole_jrev, whole_action)
            logger.debug("current Q shape: {}".format(current_Q.shape))
            logger.debug("current Q: {}".format(current_Q))

            # next actions
            next_actions = [
                self.actors_target[i](next_img_batch[:, i, :], next_info_batch[:, i, :], next_jrev_batch[:, i, :])
                for i in range(self.n_agents)]
            next_actions = th.stack(next_actions)  # list to tensor 连接
            logger.debug("learn next action: {}".format(next_actions.shape))  # torch.Size([10, 2, 4])
            next_actions = (next_actions.transpose(0, 1).contiguous())  # todo 查查contiguous()
            logger.debug("learn next action: {}".format(next_actions.shape))  # torch.Size([2, 10, 4])

            # target q
            target_Q = self.critics_target[agent](
                next_img_batch[:, agent, :],
                next_info_batch.view(-1, self.n_agents * self.n_info),
                next_jrev_batch.view(-1, self.n_agents * self.n_jrev),
                next_actions.view(-1, self.n_agents * self.n_actions)
            )
            logger.debug("target Q shape: {}".format(target_Q.shape))
            logger.debug("target Q: {}".format(target_Q))

            target_Q = (target_Q * self.GAMMA) + (
                    reward_batch[:, agent].unsqueeze(1) * self.scale_reward)
            logger.debug("reward target Q shape: {}".format(target_Q.shape))
            logger.debug("reward target Q: {}".format(target_Q))

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            # optim acitot
            self.actor_optimizer[agent].zero_grad()
            img_i = img_batch[:, agent, :]
            info_i = info_batch[:, agent, :]
            jrev_i = jrev_batch[:, agent, :]
            action_i = self.actors[agent](img_i, info_i, jrev_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](img_batch[:, agent, :], whole_info, whole_jrev, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # check to replace target parameters
        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_iter == 0:  # todo
            logger.info('\ntarget_params_replaced\n')
            for i in range(self.n_agents):
                # replace
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

                # save model
                step_counter_str = '%09d' % self.learn_step_counter
                model_name = 'model/maddpg/model_{}_agent{}.pkl'.format(step_counter_str, i)
                th.save(self.actors[i].state_dict(), model_name)

        self.learn_step_counter += 1

        logger.debug("c_loss: {}, a_loss: {}".format(c_loss, a_loss))
        return c_loss, a_loss

    def select_action(self, agent_i, img_obs, info_obs, jrev_obs):
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
        jrev_obs = th.unsqueeze(FloatTensor(jrev_obs), 0)
        # action
        action = self.actors[agent_i](img_obs, info_obs, jrev_obs).squeeze()
        logger.debug('select action: {}'.format(action))
        # 加噪声
        action += th.from_numpy(
            np.random.randn(self.n_actions) * self.var[agent_i]).type(FloatTensor)
        if self.episode_done > self.episodes_before_train and self.var[agent_i] > 0.005:  # todo 0.05
            self.var[agent_i] *= 0.999998  # 噪声稀释
        logger.debug('select action+random: {}'.format(action))
        # action = th.clamp(action, -1.0, 1.0)
        action = th.tanh(action)
        logger.debug('select action+tanh: {}'.format(action))

        # tensor to ndarray
        if self.use_cuda:
            action = action.data.cpu()  # todo 搜索tensor.data.cpu()的用法
        else:
            action = action.detach()
        action = action.numpy()

        return action
