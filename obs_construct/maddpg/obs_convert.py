import numpy as np
import copy

class ObsConvert:
    def __init__(self, detector_num, fighter_num):
        self.detector_num = detector_num
        self.fighter_num = fighter_num

        self.enermy_detector_num = detector_num
        self.enermy_fighter_num = fighter_num

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        
        detector_data, fighter_data = self.__get_data_obs_maddpg(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        
        # o方
        # 预警机
        for x in range(self.detector_num):
            data_context = detector_data[x, :]
            detector_obs_list.append(data_context)
        # 战机
        for x in range(self.fighter_num):
            data_context = fighter_data[x, :]
            fighter_obs_list.append(data_context)
        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        return obs_dict

    
    def __get_data_obs_maddpg(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        other_num = self.detector_num+self.enermy_detector_num+self.fighter_num+self.enermy_fighter_num+self.enermy_detector_num+self.enermy_fighter_num-1
        detector_data = np.full((self.detector_num, other_num*2+6), -1, dtype=np.int32)
        fighter_data = np.full((self.fighter_num, other_num*2+10), -1, dtype=np.int32)

        # Detector info
        for x in range(self.detector_num):
            detector_data[x,1] = detector_data_obs_list[x]['alive']
            detector_data[x,2] = detector_data_obs_list[x]['pos_x']
            detector_data[x,3] = detector_data_obs_list[x]['pos_y']
            detector_data[x,4] = detector_data_obs_list[x]['course']
            detector_data[x,5] = detector_data_obs_list[x]['r_iswork']
            detector_data[x,6] = detector_data_obs_list[x]['r_fre_point']
            index_n = 0
            for i in range(self.detector_num):
                if i!=x:
                    detector_data[x,6+index_n] = detector_data_obs_list[x]['pos_x'] - detector_data_obs_list[i]['pos_x']
                    index_n+=1
                    detector_data[x,6+index_n] = detector_data_obs_list[x]['pos_y'] - detector_data_obs_list[i]['pos_y']
                    index_n+=1
            for i in range(self.fighter_num):
                detector_data[x,6+index_n] = detector_data_obs_list[x]['pos_x'] - fighter_data_obs_list[i]['pos_x']
                index_n+=1
                detector_data[x,6+index_n] = detector_data_obs_list[x]['pos_y'] - fighter_data_obs_list[i]['pos_y']
                index_n+=1
            
            for i in range(self.enermy_detector_num+self.enermy_fighter_num):
                detector_data[x,6+index_n] = 0
                index_n+=1
                detector_data[x,6+index_n] = 0
                index_n+=1
            ###record the id of found enermies
            #for i in range(self.enermy_detector_num+self.enermy_fighter_num):
                #detector_data[x,6+index_n] = 0
                #index_n+=1

            disposed_enermy=[]
            temp = (self.fighter_num+self.detector_num-1)*2
            for item in detector_data_obs_list[x]['r_visible_list']:
                detector_data[x,6+temp+2*(item['id']-1)] = detector_data_obs_list[x]['pos_x'] - item['pos_x']
                detector_data[x,7+temp+2*(item['id']-1)] = detector_data_obs_list[x]['pos_y'] - item['pos_y']
                disposed_enermy.append(item['id']-1)
            
            #temp = (self.fighter_num+self.detector_num-1)*2+2*(self.enermy_detector_num+self.enermy_fighter_num)
            #for item in detector_data_obs_list[x]['r_visible_list']:
                #detector_data[x,6+temp+item['id']-1] = 1

            
             
        # Fighter info        
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                break
            fighter_data[x,0] = fighter_data_obs_list[x]['alive']
            fighter_data[x,1] = fighter_data_obs_list[x]['pos_x']
            fighter_data[x,2] = fighter_data_obs_list[x]['pos_y']
            fighter_data[x,3] = fighter_data_obs_list[x]['course']
            fighter_data[x,4] = fighter_data_obs_list[x]['r_iswork']
            fighter_data[x,5] = fighter_data_obs_list[x]['r_fre_point']
            fighter_data[x,6] = fighter_data_obs_list[x]['j_iswork']
            fighter_data[x,7] = fighter_data_obs_list[x]['j_fre_point']
            fighter_data[x,8] = fighter_data_obs_list[x]['l_missile_left']
            fighter_data[x,9] = fighter_data_obs_list[x]['s_missile_left']

            ####observation of teammater
            index_n = 0
            for i in range(self.detector_num):
                fighter_data[x,10+index_n] = fighter_data_obs_list[x]['pos_x'] - detector_data_obs_list[i]['pos_x']
                index_n+=1
                fighter_data[x,10+index_n] = fighter_data_obs_list[x]['pos_y'] - detector_data_obs_list[i]['pos_y']
                index_n+=1
            for i in range(self.fighter_num):
                if i!=x and fighter_data_obs_list[i]['alive']:
                    fighter_data[x,10+index_n] = fighter_data_obs_list[x]['pos_x'] - fighter_data_obs_list[i]['pos_x']
                    index_n+=1
                    fighter_data[x,10+index_n] = fighter_data_obs_list[x]['pos_y'] - fighter_data_obs_list[i]['pos_y']
                    index_n+=1
            
            ####observation of enermy
            for i in range(self.enermy_detector_num+self.enermy_fighter_num):
                fighter_data[x,10+index_n] = 0
                index_n+=1
                fighter_data[x,10+index_n] = 0
                index_n+=1
            
            disposed_enermy=[]
            temp = (self.fighter_num+self.detector_num-1)*2
            for item in fighter_data_obs_list[x]['r_visible_list']:
                fighter_data[x,10+temp+2*(item['id']-1)] = fighter_data_obs_list[x]['pos_x'] - item['pos_x']
                fighter_data[x,11+temp+2*(item['id']-1)] = fighter_data_obs_list[x]['pos_y'] - item['pos_y']
                disposed_enermy.append(item['id']-1)
            
            for item in joint_data_obs_dict['passive_detection_enemy_list']:
                 if item['id']-1 not in disposed_enermy:
                    fighter_data[x,10+temp+2*(item['id']-1)] = fighter_data_obs_list[x]['pos_x'] - item['pos_x']
                    fighter_data[x,11+temp+2*(item['id']-1)] = fighter_data_obs_list[x]['pos_y'] - item['pos_y']
                    disposed_enermy.append(item['id']-1)                
            
            for item in range(self.enermy_detector_num+self.enermy_fighter_num):
                fighter_data[x,10+index_n] = 0
                index_n+=1
                fighter_data[x,10+index_n] = 0
                index_n+=1
            temp = (self.fighter_num+self.detector_num-1+self.enermy_detector_num+self.enermy_fighter_num)*2
            disposed_enermy=[]
            for item in fighter_data_obs_list[x]['j_recv_list']:
                if item['id']-1 not in disposed_enermy:
                    fighter_data[x,10+temp+2*(item['id']-1)] =  item['direction']
                    fighter_data[x,11+temp+2*(item['id']-1)] =  item['r_fp']
                    disposed_enermy.append(item['id']-1)
            
        
        return detector_data, fighter_data
    
    def information_integration(self,obs_raw_dict):
        other_radar_info = []
        other_disturb_info = []
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        for x in range(self.fighter_num):
            disposed_enermy=[]
            for item in fighter_data_obs_list[x]['r_visible_list']:
                if item['id'] not in disposed_enermy:
                    disposed_enermy.append(item['id'])
                    other_radar_info.append(item)
            for item in fighter_data_obs_list[x]['j_recv_list']:
                if item['id'] not in disposed_enermy:
                    disposed_enermy.append(item['id'])
                    other_disturb_info.append(item)
        return other_radar_info,other_disturb_info
    
    def update_obs_raw_dict(self,obs_raw_dict,other_radar_info,other_disturb_info):
        new_obs_raw_dict = copy.deepcopy(obs_raw_dict)
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        for x in range(self.fighter_num):
            own_radar_id = []
            own_distub_id = []
            for item in fighter_data_obs_list[x]['r_visible_list']:
                own_radar_id.append(item['id'])
            for item in fighter_data_obs_list[x]['j_recv_list']:
                own_distub_id.append(item['id'])
        
            for item in other_radar_info:
                if item['id'] not in own_radar_id:
                    new_obs_raw_dict['fighter_obs_list'][x]['r_visible_list'].append(item)
            #for item in other_disturb_info:
                #if item['id'] not in own_distub_id:
                    #new_obs_raw_dict['fighter_obs_list'][x]['j_recv_list'].append(item)
        return  new_obs_raw_dict
    
    def liu_rule_obs_to_train_obs(self,blue_obs_dict_raw):
        obs_dict = {}
        fighter_data_obs_list=[]
        #fighter_data_obs_list = blue_obs_dict_raw.values()
        for key,value in blue_obs_dict_raw.items():
            if not (key == 'strike_list'):
                fighter_data_obs_list.append(value)
        
        detector_data, fighter_data = self.__get_data_obs_maddpg([], fighter_data_obs_list, {'passive_detection_enemy_list':[]})

        # 战机
        fighter_obs_list = []
        for x in range(self.fighter_num):
            data_context = fighter_data[x, :]
            fighter_obs_list.append(data_context)
        obs_dict['detector'] = []
        obs_dict['fighter'] = fighter_obs_list
        return obs_dict
        



