import json
import os
import re
import shutil

def json_oprate(path):
    '''

    :param path: json文件path
    :return: 返回json的内容
    '''
    with open(path,'r',encoding='utf8') as f:
        if re.match('.*PASI 0.json',path):
            return {'head': {'area': '0', 'desquamation': '0', 'erythema': '0', 'infiltration': '0'},
                    'torso': {'area': '0', 'desquamation': '0', 'erythema': '0', 'infiltration': '0'},
                    'upperLimb': {'area': '0', 'desquamation': '0', 'erythema': '0', 'infiltration': '0'},
                    'lowerLimb': {'area': '0', 'desquamation': '0', 'erythema': '0', 'infiltration': '0'}}
        json_data=json.load(f)
        return json_data

def add_string(path,json_data,part_str,body_part):
    if part_str in json_data:
        # path在这变了
        desquamation_grade=json_data[part_str]['desquamation']
        ill_dict['desquamation'][int(desquamation_grade)].append(path+os.sep+body_part)
        erythema_grade=json_data[part_str]['erythema']
        ill_dict['erythema'][int(erythema_grade)].append(path+os.sep+body_part)
        infiltration_grade=json_data[part_str]['infiltration']
        ill_dict['infiltration'][int(infiltration_grade)].append(path+os.sep+body_part)
        area_grade=json_data[part_str]['area']
        ill_dict['area'][int(area_grade)].append(path+os.sep+body_part)


def find_json(path):
    '''
    :param path: 文件夹路径
    :return: 返回json文件path【一个】
    '''
    ###################################################################这里
    name=os.listdir(path+os.sep)
    for each in name:
        if re.match('.*\.json$',each):
            return path+os.sep+each


def get_all_person(path):
    '''

    :param path: 数据集路径
    :return: 返回一个ill_dict,记录不同症状不同level的path
    '''
    # 得到人的dirs
    dir_name=[path+os.sep+each for each in os.listdir(path) if os.path.isdir(path+os.sep+each) ]
    # 得到人的子dir
    data_dir_name=[]
    for dir in dir_name:
        data_dir_name.extend([dir+os.sep+each for each in os.listdir(dir) if os.path.isdir(dir+os.sep+each)])
    print(data_dir_name)
    for each in data_dir_name:
        '''
        遍历所有的dir中的json文件，获得一个ill_dict
        each:文件夹路径
        '''
        # json_file_name没了
        json_file_name=find_json(each)
        #print(json_file_name)
        json_data=json_oprate(json_file_name)
        add_string(each,json_data,'head','头部')
        add_string(each,json_data,'torso','躯干')
        add_string(each,json_data,'upperLimb','上肢')
        add_string(each,json_data,'lowerLimb','下肢')
    print(ill_dict)
    return  ill_dict

def file_move(ill_dict):
    # items()
    for each in ill_dict.items():
        ill_symptom=each[0]
        ill_info=each[1]
        read_and_move(ill_symptom,ill_info)
        # print(ill_symptom)
        # print(ill_info)
        # print('finish!')

def read_and_move(ill_symptom,ill_info):
    '''
    把一个症状的病，复制到一个symptom文件夹中
    :param ill_symptom: 'desquamation'
    :param ill_info: dict:{0: ['E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部'], 1: ['E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢'], 2: ['E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部'], 3: ['E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\头部', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\躯干', 'E:/post-graduate/data/pasi/李/2020-07-23\\上肢', 'E:/post-graduate/data/pasi/李/2020-07-23\\下肢'], 4: ['E:/post-graduate/data/pasi/李/2020-07-23\\下肢'], 5: []}
    :return:
    '''
    path=r'E:/post-graduate/data/total'
    for i in range(len(ill_info)):
        ill_dir_list=ill_info[i]
        for ill_dir in ill_dir_list:
            copy_all_file(ill_dir,path+os.sep+ill_symptom+os.sep+str(i))



def copy_all_file(path_from,path_des):
    '''
    把一个症状的一个等级的病复制到一个文件夹中
    :param path_from: 一个list,保存所有的相同的symphony[i]
    :param path_des:
    :return:
    '''
    # 这个得到的是file名字
    if not os.path.exists(path_from):
        return
    files=[path_from+os.sep+each for each in os.listdir(path_from)]
    for each in files:
        # print(each)
        if not os.path.exists(path_des):
            os.mkdir(path_des)
        if os.path.exists(path_from):
            print(each)
            shutil.copy(each, path_des)

    print("!!!!!!!!!!!finish one dir!!!!!!!!!!!!!!")



if __name__=='__main__':
    path=r'E:/post-graduate/data/pasi'
    ill_dict={'desquamation':{0:[],1:[],2:[],3:[],4:[],5:[],6:[]},
              'erythema':{0:[],1:[],2:[],3:[],4:[],5:[],6:[]},
              'infiltration':{0:[],1:[],2:[],3:[],4:[],5:[],6:[]},
              'area':{0:[],1:[],2:[],3:[],4:[],5:[],6:[]},
              }
    ill_dict=get_all_person(r'E:/post-graduate/data/report')
    file_move(ill_dict)
    #print(ill_dict)
    # with open('E:/post-graduate/data/a.json','w',encoding='utf8') as f:
    #     json.dump(ill_dict,f)