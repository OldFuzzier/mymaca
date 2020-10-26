#
# coding=utf-8

import numpy as np
import copy

from config import logger


class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.img_obs_reduce_ratio = 10

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        # 所有敌方单位位置(探测单元观察 2x100x100x3矩阵 战斗单元观察　10x100x100x3矩阵)
        # 所有全局信息可见敌方单元位置( 被动探测＋探测者＋战斗者　1x100x100x2矩阵)
        detector_img, fighter_img, joint_img = self.__get_img_obs(detector_data_obs_list, fighter_data_obs_list,
                                                                  joint_data_obs_dict)
        # 当前航向　　2x1矩阵　　当前航向，剩余远程载弹量，剩余近程载弹量　１０x3矩阵
        # 侦测单元位置 2x100x100  战斗单元位置　10x100x100　
        # detector_data, fighter_data,detector_pos ,fighter_pos = self.__get_data_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        detector_data, fighter_data = self.__get_data_obs(detector_data_obs_list,
                                                          fighter_data_obs_list,
                                                          joint_data_obs_dict)

        detector_jvisible, fighter_jvisible = self.__get_jvisible_obs(detector_data_obs_list, fighter_data_obs_list,
                                                                      joint_data_obs_dict)

        # 探测单元２和战斗单元１０存活状态　１２x1矩阵
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)
        # o方
        # 预警机
        for x in range(self.detector_num):
            img_context = detector_img[x, :, :, :]
            # 2x100x100x3矩阵　和　1x100x100x2矩阵　合并　c.shape[２] = a.shape[２]+b.shape[２]
            # 100x100x3矩阵　和　100x100x2矩阵　合并后变为100x100x５矩阵
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = detector_data[x, :]
            j_visible_context = detector_jvisible[x, :]
            detector_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                                      'alive': alive_status[x][0], 'j_visible': copy.deepcopy(j_visible_context)})
        # 战机
        for x in range(self.fighter_num):
            img_context = fighter_img[x, :, :, :]
            # 10x100x100x3矩阵 和　1x100x100x2矩阵　合并　c.shape[２] = a.shape[２]+b.shape[２]
            # 　100x100x3矩阵 和　100x100x2矩阵　合并后变为100x100x５矩阵
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = fighter_data[x, :]
            j_visible_context = fighter_jvisible[x, :]
            fighter_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                                     'alive': alive_status[x + self.detector_num][0],
                                     'j_visible': copy.deepcopy(j_visible_context)})
        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        return obs_dict

    def __get_alive_status(self, detector_data_obs_list, fighter_data_obs_list):
        alive_status = np.full((self.detector_num + self.fighter_num, 1), True)
        # 探测单元　存活　从０　开始计数
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        # 战斗单元　存活　从０＋探测单元总数　开始计数
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x + self.detector_num][0] = False
        return alive_status

    def __get_img_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # logger.debug("into __get_img_obs", "=" * 30)
        # logger.debug("into __get_img_obs", "=" * 30)
        # logger.debug("into __get_img_obs", "=" * 30)
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 个体img：所有敌方＋己方单位位置
        # 2x100x100x3矩阵　默认值０　表示两个探测单元观察的100x100矩阵的三个值
        detector_img = np.full((self.detector_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        # 10x100x100x3矩阵 　默认值０　表示十个战斗单元观察的100x100矩阵的三个值
        fighter_img = np.full((self.fighter_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        # 企鹅据img：所有可见敌方单元位置和类型
        # 1x100x100x2矩阵  默认值０
        joint_img = np.full((1, img_obs_size_x, img_obs_size_y, 2), 0, dtype=np.int32)

        # set all self unit pos, detector: 1, fighter: 2, self: 255
        # 设置100x100矩阵默认值为０
        tmp_pos_obs = np.full((img_obs_size_x, img_obs_size_y), 0, dtype=np.int32)
        # 填充己方侦查者位置１
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 1)
        # 填充己方战斗者值２
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 2)

        # Detector obs
        for x in range(self.detector_num):
            # if not alive, skip
            if not detector_data_obs_list[x]['alive']:
                continue
            # r_visible_list
            # 雷达观测到的敌方单位列表,每一元素字典,表示敌方单位信息,结构为
            # {‘id’:编号,’type’:类型(0:探测单元,1:攻击单元) ,‘pos_x’:
            # 横向坐标,’pos_y’:纵向坐标}
            # self detection target. target: id
            # if len(detector_data_obs_list[x]['r_visible_list'])>0:
            #     logger.debug("detector_data_obs_list=",len(detector_data_obs_list[x]['r_visible_list']))
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                # 每一个探测者看到的[x, :, :, 0]矩阵中某个点（对象）对应一个ｉｄ
                # logger.debug("探测者看到的", detector_data_obs_list[x]['r_visible_list'][y][
                #     'pos_y'], detector_data_obs_list[x]['r_visible_list'][y][
                #           'pos_x'])
                self.__set_value_in_img(detector_img[x, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])
            # self detection target. target: type (detector: 1, fighter 2)
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                # 每一个探测者看到的[x, :, :, １]矩阵中某个点（对象）对应一个ｔｙｐｅ
                self.__set_value_in_img(detector_img[x, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
            # friendly pos. self: 255, other: type (detector: 1, fighter 2)
            detector_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            # 每一个探测者看到的[x, :, :, ２]矩阵中自己（对象）对应一个２５５，其它为己方探测者１，己方战斗者２
            self.__set_value_in_img(detector_img[x, :, :, 2],
                                    int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # Fighter obs
        for x in range(self.fighter_num):
            # if not alive, skip
            if not fighter_data_obs_list[x]['alive']:
                continue
            # r_visible_list
            # 雷达观测到的敌方单位列表,每一元素字典,表示敌方单位信息,结构为
            # {‘id’: 编 号 ,’type’: 类 型 (0: 探 测 单 元 , 1 : 攻 击 单 元 ) , ‘pos_x’: 横 向 坐
            # 标,’pos_y’:纵向坐标}
            # self detection target. target: id
            # if len(fighter_data_obs_list[x]['r_visible_list'])>0:
            #     logger.debug("fighter_data_obs_list=" , len(fighter_data_obs_list[x]['r_visible_list']))
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                # 每一个战斗者看到的[x, :, :, 0]矩阵中某个点（对象）对应一个ｉｄ
                # logger.debug("战斗者看到的", x, y, fighter_data_obs_list[x]['r_visible_list'][y][
                #     'pos_y'], fighter_data_obs_list[x]['r_visible_list'][y][
                #           'pos_x'])
                self.__set_value_in_img(fighter_img[x, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])
            # self detection target. target: type (detector: 1, fighter 2)
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                # 每一个战斗者看到的[x, :, :, １]矩阵中某个点（对象）对应一个ｔｙｐｅ1/2
                self.__set_value_in_img(fighter_img[x, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
            # friendly pos. self: 255, other: type (detector: 1, fighter 2)
            fighter_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            # 每一个战斗者看到的[x, :, :, ２]矩阵中自己（对象）对应一个２５５，其它为己方探测者１，己方战斗者２
            self.__set_value_in_img(fighter_img[x, :, :, 2],
                                    int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # Global obs
        # Passive detection　 被动探
        # if len(joint_data_obs_dict['passive_detection_enemy_list'])>0:
        #     logger.debug("joint_data_obs_dict=" , len(joint_data_obs_dict['passive_detection_enemy_list']))
        for x in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
            # Channel: detected enemy pos. value=enemy id
            self.__set_value_in_img(joint_img[0, :, :, 0], int(
                joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x][
                                            'pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['id'])
            # Channe2: detected enemy pos. value=enemy type
            self.__set_value_in_img(joint_img[0, :, :, 1], int(
                joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x][
                                            'pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['type'] + 1)
        # detector　探测
        for x in range(self.detector_num):
            # if len(detector_data_obs_list[x]['r_visible_list'])>0:
            #     logger.debug("joint_img detector_data_obs_list=", len(detector_data_obs_list[x]['r_visible_list']))
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                # Channel: detected enemy pos. value=enemy id
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])
                # Channe2: detected enemy pos. value=enemy type
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        # fighter
        for x in range(self.fighter_num):
            # if  len(fighter_data_obs_list[x]['r_visible_list'])>0:
            #     logger.debug("joint_img fighter_data_obs_list=" , len(fighter_data_obs_list[x]['r_visible_list']))
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                # Channel: detected enemy pos. value=enemy id
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])
                # Channe2: detected enemy pos. value=enemy type
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        return detector_img, fighter_img, joint_img

    def __set_value_in_img(self, img, pos_x, pos_y, value):
        """
        draw 3*3 rectangle in img
        :param img:
        :param pos_x:
        :param pos_y:
        :param value:
        :return:
        """
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 左上角
        if pos_x == 0 and pos_y == 0:
            img[pos_x: pos_x + 2, pos_y: pos_y + 2] = value
        # 左下角
        elif pos_x == 0 and pos_y == (img_obs_size_y - 1):
            img[pos_x: pos_x + 2, pos_y - 1: pos_y + 1] = value
        # 右上角
        elif pos_x == (img_obs_size_x - 1) and pos_y == 0:
            img[pos_x - 1: pos_x + 1, pos_y: pos_y + 2] = value
        # 右下角
        elif pos_x == (img_obs_size_x - 1) and pos_y == (img_obs_size_y - 1):
            img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 1] = value
        # 左边
        elif pos_x == 0:
            img[pos_x: pos_x + 2, pos_y - 1: pos_y + 2] = value
        # 右边
        elif pos_x == img_obs_size_x - 1:
            img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 2] = value
        # 上边
        elif pos_y == 0:
            img[pos_x - 1: pos_x + 2, pos_y: pos_y + 2] = value
        # 下边
        elif pos_y == img_obs_size_y - 1:
            img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 1] = value
        # 其他位置
        else:
            img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 2] = value

    def __get_data_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        # img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 2x1矩阵　默认值－１
        detector_data = np.full((self.detector_num, 1), -1, dtype=np.int32)
        # detector_pos = np.full((self.detector_num, img_obs_size_x, img_obs_size_y), 0, dtype=np.int32)
        # 10x3矩阵　默认值－１
        fighter_data = np.full((self.fighter_num, 3), -1, dtype=np.int32)
        # fighter_pos = np.full((self.fighter_num, img_obs_size_x, img_obs_size_y), 0, dtype=np.int32)
        # Detector info
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                # course 航向(0-359),水平向右为 0 度,顺时针旋转
                detector_data[x, 0] = detector_data_obs_list[x]['course']
                # self.__set_value_in_img(detector_pos[x, :, :],
                #                     int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                #                     int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio),1)
        # Fighter info
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                # course 航向(0-359),水平向右为 0 度,顺时针旋转
                fighter_data[x, 0] = fighter_data_obs_list[x]['course']
                # l_missle_left 远程导弹剩余数目
                fighter_data[x, 1] = fighter_data_obs_list[x]['l_missile_left']
                # s_missle_left 短程导弹剩余数目
                fighter_data[x, 2] = fighter_data_obs_list[x]['s_missile_left']
                # # ---------------radar_point
                # fighter_data[x, 3] = fighter_data_obs_list[x]['r_fre_point']
                # # -------------------jamming_point
                # fighter_data[x, 4] = fighter_data_obs_list[x]['j_fre_point']
                # fighter_data[x, 3] = fighter_data_obs_list[x]['last_reward']
        # return detector_data, fighter_data ,detector_pos ,fighter_pos
        return detector_data, fighter_data

    # def __set_value_in_img(self, jvisible, direction, value):
    #     jvisible[direction:]

    def __get_jvisible_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):

        fighter_jvisible_data = np.full((self.fighter_num, self.fighter_num, 4), -1, dtype=np.int32)  # (10, 10, 4)
        detector_jvisible_data = np.full((self.detector_num, self.detector_num, 4), -1, dtype=np.int32)

        for x in range(self.fighter_num):
            # 10*4的被动观测信息
            if fighter_data_obs_list[x]['alive']:
                for y in range(len(fighter_data_obs_list[x]['j_recv_list'])):
                    recv_id = fighter_data_obs_list[x]['j_recv_list'][y]['id']
                    fighter_jvisible_data[x, recv_id-1, 0] = fighter_data_obs_list[x]['j_recv_list'][y]['id']
                    fighter_jvisible_data[x, recv_id-1, 1] = fighter_data_obs_list[x]['j_recv_list'][y]['type']
                    fighter_jvisible_data[x, recv_id-1, 2] = fighter_data_obs_list[x]['j_recv_list'][y]['direction']  # 敌人相对自己的位置
                    fighter_jvisible_data[x, recv_id-1, 3] = fighter_data_obs_list[x]['j_recv_list'][y]['r_fp']  # 雷达频点

        return detector_jvisible_data, fighter_jvisible_data
