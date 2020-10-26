import math
import random
import numpy as np


def getGroup(num):
    '''
       首先，攻击单元随机分组，
       最多三层，
               一层队列（最多一组5个，随机２－5，不可少于一个）　
               二层队列（第一层随机１－２，第二层最多一组４个，随机２－４，不可少于一个）
               三层队列（第一层最多１个，第二层先取剩下的一半，随机２－３，不可少于一个，第三层为剩下的）　
               间隔距离ｙ　４０　ｘ　１０
    '''
    group = []
    if num>8:
        layer = random.randint(1, 3)
    elif 5<=num<=8:
        layer = random.randint(1, 2)
    elif num<5:
        layer = 1
    #print("getGroup b", layer,num)
    if layer==1:
        group=getGroup1(num,group,5)
    elif layer==2:
        group=getGroup2(num,group)
    elif layer==3:
        group=getGroup3(num,group)

    # 每一个单元在100x100矩阵上表现为3x3矩阵
    # 每一组排列上下间隔３
    grouptmp = []
    if layer == 1:
        lg = len(group)
        for i in range(lg):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg + 1)) - 3 * lg - 2, 96])
    elif layer == 2:
        lg1 = len(group[0])
        lg2 = len(group[1])
        for i in range(lg1):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg1 + 1)) - 3 * lg1 - 2, 96])
        for i in range(lg2):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg2 + 1)) - 3 * lg2 - 2, 88])
    elif layer == 3:
        lg1 = len(group[0])
        lg2 = len(group[1])
        lg3 = len(group[2])
        for i in range(lg1):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg1 + 1)) - 3 * lg1 - 2, 96])
        for i in range(lg2):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg2 + 1)) - 3 * lg2 - 2, 88])
        for i in range(lg3):
            grouptmp.append([math.ceil(100 * (i + 1) / (lg3 + 1)) - 3 * lg3 - 2, 80])

    # 默认设置三组334
    layer = 1
    group = [3, 3, 4]
    grouptmp = [[10, 90], [45, 90], [90, 90]]
    return layer,group,grouptmp

def getGroup1(num,group,max):
    if num > (max+2):
        tmp = random.randint(2, max)
        group.append(tmp)
        #print("getGroup1-1b", num-tmp, group, max)
        if num-tmp>0:
            group=getGroup1(num-tmp,group,max)
    elif num < 3:
        group.append(num)
    elif 3 <= num <= (max+2):
        if num-2>2:
            tmp = random.randint(2, num-2)
        else:
            tmp = num
        group.append(tmp)
        if 0<(num-tmp)<3:
            group.append(num-tmp)
        elif (num-tmp)>=3:
            #print("getGroup1-2b", num - tmp, group, max)
            group=getGroup1(num - tmp, group,max)
    #print("getGroup1-end", num, group,max)
    return group

def getGroup2(num,group):
    lay1=random.randint(1, 2)
    group.append([lay1])
    lay2sum=num-lay1
    #print("getGroup2", num, lay2sum)
    group.append(getGroup1(lay2sum,[],4))
    #print("getGroup2", num, group)
    return group

def getGroup3(num,group):
    lay1 = random.randint(1, 2)
    group.append([lay1])
    lay2sum = math.ceil((num - lay1) / 2)
    #print("getGroup3 b1", num, lay2sum)
    group.append(getGroup1(lay2sum, [], 3))
    lay3sum = num - lay1 - lay2sum
    #print("getGroup3 b1", num, lay3sum)
    group.append(getGroup1(lay3sum, [], 3))
    #print("getGroup3", num, group)
    return group

def getarc360(arc):
    if 0<=arc<=180:
        return arc
    elif -180<=arc<0:
        return 360+arc
    # if 0<=arc<=180:
    #     return 360-arc
    # elif -180<=arc<0:
    #     return -arc

#tmp_info_obs 024 表示航向０远程导弹２中程导弹４
def getGroupposition(num,layer,group,grouptmp,red_obs_dict_fighter,fighter_model,preposs):
    true_action_list=[]
    tmp_action_list=[]

    #判断是否全局侦测到可攻击的对象      tmp_img_obs[3][r][c]=id tmp_img_obs[4][r][c]=type
    #garr=np.where(tmp_img_obs[3] > 0)
    ifrevease = False
    #判断由那个小队来攻击敌人
    groupcount = 1
    for y in range(num):
        tmp_img_obs = red_obs_dict_fighter[y]['screen']
        tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
        tmp_info_obs = red_obs_dict_fighter[y]['info']
        # 判断是否战斗单元侦测到可攻击的对象   tmp_img_obs[0][r][c]=id tmp_img_obs[1][r][c]=type
        farr=np.where(tmp_img_obs[0] > 0)

        if red_obs_dict_fighter[y]['alive']:
            # 判断位置值为255 的自己位置 oarr[0][4] oarr[1][4]
            oarr = np.where(tmp_img_obs[2] == 255)
            curpos=[oarr[0][int((len(oarr[0])-1)/2)],oarr[1][int((len(oarr[1])-1)/2)]]
            # tmp_action = fighter_model.choose_action(tmp_img_obs, tmp_info_obs)
            tmp_action=[0]
            fightpos=[0,0]
            if len(farr[0])>0:
                fightpos = [farr[0][int((len(farr[0]) - 1) / 2)], farr[1][int((len(farr[1]) - 1) / 2)]]
                tmp_action = [tmp_img_obs[0][farr[0][0]][farr[1][0]]]
                print("敌方位置",fightpos[1],fightpos[0],tmp_action,y)
                # 判断是否存在挂弹为０ 看到敌人，反向
                if tmp_info_obs[1] == 0 and tmp_info_obs[2] == 0:
                    ifrevease = True

            true_action,tmp_action,preposs[y]=getOneposition(layer,group,grouptmp,groupcount,fightpos,curpos,tmp_action,tmp_info_obs,preposs[y])
            true_action_list.append(true_action)
            #tmp_action存储攻击的敌方单元id,原则每一个被最近一组攻击两次
            tmp_action_list.append(tmp_action)
            groupcount += 1
        else:
            true_action_list.append(np.array([0, 0, 0, 0], dtype=np.int32))
            tmp_action_list.append([0])



    return true_action_list,tmp_action_list,preposs

def judgecourse(curpos,fightpos,prepos,true_action0,issee):
    # 计算航向 w
    if 10 < curpos[1] < 90:
        if prepos[2] == 0:
            prepos[0] = prepos[0] + 3
        else:
            prepos[0] = prepos[0] - 3
        true_action0 = prepos[2]
    elif curpos[1] >= 90:
        true_action0 = 180
        prepos[0] = prepos[0] - 3
    elif curpos[1] <= 10:
        true_action0 = 0
        prepos[0] = prepos[0] + 3

    if curpos[0] == 0:
        true_action0 = 120
        prepos[0] = prepos[0] - 3
    elif curpos[0] == 98:
        true_action0 = 330
        prepos[0] = prepos[0] + 3
    #看到敌人，只有中程导弹，那么航向对着敌人开去
    if issee and prepos[3]>3:
        true_action0 = getarc360(int(math.atan2(fightpos[0] - curpos[0], fightpos[1] - curpos[1])*180/math.pi))
    # if 10 < curpos[1] < 90:
    #     if prepos[1] == 0:
    #         prepos[1] = 60
    #         prepos[0] = prepos[0] + 1.5
    #     elif prepos[1] == 60:
    #         prepos[1] = 300
    #         prepos[0] = prepos[0] + 1.5
    #     elif prepos[1] == 300:
    #         prepos[1] = 60
    #         prepos[0] = prepos[0] + 1.5
    #     elif prepos[1] == 120:
    #         prepos[1] = 240
    #         prepos[0] = prepos[0] - 1.5
    #     elif prepos[1] == 240:
    #         prepos[1] = 120
    #         prepos[0] = prepos[0] - 1.5
    #     elif prepos[1] == 180:
    #         prepos[1] = 120
    #         prepos[0] = prepos[0] - 1.5
    #     true_action[0] = prepos[1]
    # elif curpos[1] >= 90:
    #     true_action[0] = 180
    #     prepos[0] = prepos[0] - 3
    # elif curpos[1] <= 10:
    #     true_action[0] = 0
    #     prepos[0] = prepos[0] + 3
    return prepos[0],true_action0

def getOneposition(layer,group,grouptmp,groupcount,fightpos,curpos,tmp_action,tmp_info_obs,prepos):
    true_action = np.array([0, 1, 0, 0], dtype=np.int32)
    curnum = 0
    curg = 0
    for l in range(layer):
        if layer == 1:
            layerr = group
        else:
            layerr = group[l]
        for i in range(len(layerr)):
            layerrc = layerr[i]
            # print("if curnum < groupcount <= (curnum+layerrc):",curnum,groupcount,layerrc)
            if curnum < groupcount <= (curnum + layerrc):
                # print("curg",curg)
                # pos = grouptmp[curg]
                # posx = pos[1]
                # posy = pos[0]
                # 如果未达｜已过｜屏幕中间　４５　９０　如果等于９０　反向
                # if curpos[1] < 45:
                #     posx = 45
                # elif 45 <= curpos[1] < 90:
                #     posx = 90
                # elif curpos[1] == 90:
                #     posx = 10
                # 判断剩余弹量
                print("航向-远程弹量-中程弹量", groupcount ,tmp_info_obs[0],tmp_info_obs[1],tmp_info_obs[2])
                print("位置",curpos[1],curpos[0])
                # 计算是否攻击
                # 计算雷达是否开关机
                if (prepos[3] > 1):
                    start = prepos[3] + 1 if prepos[3] < 9 else 1
                    true_action[1] = random.randint(start, 10)
                    true_action[2] = random.randint(start, 10)
                else:
                    true_action[1] = 1
                    true_action[2] = 1
                # # 此处判断弹数量耗尽的先返航
                # if tmp_info_obs[1] == 0 and tmp_info_obs[2] == 0:
                #     true_action[0] = 180
                # 没有看到敌人
                if fightpos[0] == 0:
                    print("没有看到敌人")
                    # 计算航向 w
                    prepos[0],true_action[0]=judgecourse(curpos, fightpos,prepos, true_action[0],0)
                    #true_action[0] = getarc360(int(math.atan2(posy - curpos[0], posx - curpos[1])*180/math.pi))
                    #true_action[0] = getarc360(math.atan2(posy - curpos[0], posx - curpos[1]))
                    true_action[3] = 0
                    tmp_action = [0]
                # 看到敌人
                else:
                    print("看到敌人"*10)
                    prepos[0],true_action[0]=judgecourse(curpos, fightpos,prepos, true_action[0],1)
                    #true_action[0] = getarc360(int(math.atan2(posy - fightpos[0], posx - fightpos[1])*180/math.pi))
                    #true_action[0] = getarc360(math.atan2(posy - curpos[0], posx - curpos[1]))
                    # 判断模型得到的攻击目标ｉｄ
                    if tmp_action[0] == 0:
                        print("不攻击")
                    elif 1 <= tmp_action[0] <= 2:
                        print("远程攻击敌方侦测单元",tmp_action[0])
                    elif 3 <= tmp_action[0] <= 12:
                        print("远程攻击敌方战斗单元",tmp_action[0])
                    elif 13 <= tmp_action[0] <= 14:
                        print("中程攻击敌方侦测单元",tmp_action[0])
                    elif 15 <= tmp_action[0] <= 24:
                        print("中程攻击敌方战斗单元",tmp_action[0])

                    # X 第一波攻击，远程弹打一发，然后都是中程弹
                    #if 0<tmp_info_obs[1] or (tmp_info_obs[2]<2 and tmp_info_obs[1]==1):
                    if tmp_info_obs[1]>0 :
                        prepos[3] += 1
                        taction = tmp_action[0]
                    elif tmp_info_obs[2]>0 :
                        prepos[3] += 1
                        taction = tmp_action[0] + 10
                    else:
                        taction = 0
                        # 逃跑
                        true_action[0] = true_action[0] + 180
                    # 计算航向
                    true_action[3] = taction
            curnum += layerrc
            curg += 1
    #print("true_action",curpos,true_action)
    return true_action,tmp_action,[prepos[0],prepos[1],true_action[0],prepos[3]]
