import datetime
import scipy as sp
from keras import backend as K


def my_logloss(act, pred):
    epsilon = 1e-15
    pred = K.maximum(epsilon, pred)
    pred = K.minimum(1 - epsilon, pred)
    ll = K.sum(act * K.log(pred) + (1 - act) * K.log(1 - pred))
    ll = ll * -1.0 / K.shape(act)[0]
    return ll


def logloss(act, pred):
    '''
    官方给的损失函数
    :param act: 
    :param pred: 
    :return: 
    '''
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


def get_how_much_time(time_str, year_month='2017-01', start_date_time='2017-01-010000'):
    """
    通过输入xxxxxx格式的时间，得到一个时间差。单位是秒
    """
    t_str = year_month + "-" + time_str
    t1 = datetime.datetime.strptime(t_str, '%Y-%m-%d%H%M')
    t2 = datetime.datetime.strptime(start_date_time, '%Y-%m-%d%H%M')
    how_long = t1.timestamp() - t2.timestamp()
    return how_long


def get_ad_info(ad_dict, app_categories_dict, creativeID):
    '''
    获取广告的相关信息
    :param ad_dict: 
    :param creativeID: 
    :return: 
    '''
    position = ad_dict[creativeID]
    words = position.split(',')
    adID = int(words[1])
    camgaignID = int(words[2])  # 推广计划是广告的集合，类似电脑文件夹功能
    advertiserID = int(words[3])  # 广告主id

    def get_app_categories(app_categories_dict, appID):  # 获取广告类别目录
        app_categories = app_categories_dict[appID]
        if app_categories == '0':
            return [0, 00]
        elif len(app_categories) == 1:
            return [int(app_categories), 00]
        elif len(app_categories) == 3:
            # print("app类别:", app_categories)
            # print("len", len(app_categories))
            return [int(app_categories[0]), int(str(app_categories[1] + app_categories[2]))]
        else:
            raise Exception('类别解析存在bug，请审查重新编写')

    appID = int(words[4])
    app_categories = get_app_categories(app_categories_dict, appID)  # 广告类别
    appPlatform = int(words[5])  # app 平台系统，入如苹果、安卓等

    return [camgaignID, advertiserID] + app_categories + [appPlatform]


def get_position_info(position_dict, positionID):
    '''
    根据positionid获取广告位置信息
    :param position_dict: 
    :param positionID: 
    :return: [sitesetID, positionID]
    '''
    position = position_dict[positionID]
    words = position.split(',')
    # 广告位类型
    positionType = int(words[2])
    # 站点集合ID
    sitesetID = int(words[1])
    return [sitesetID, positionID]


def get_user_info(user_dict, userID):
    '''
    根据输入的userid获取user相关的信息
    :param user_dict: 
    :param userID: 
    :return:  [age, gender, education, marriageStatus, haveBaby, hometown, residence]
    '''
    user = user_dict[userID]
    words = user.split(',')
    # print(words[0], userID)
    # 处理年龄
    if int(words[1]) < 18:
        age = 1
    elif 18 <= int(words[1]) < 25:
        age = 2
    elif 25 <= int(words[1]) < 30:
        age = 3
    elif 30 <= int(words[1]) < 35:
        age = 2
    elif 35 <= int(words[1]) < 40:
        age = 2
    elif 40 <= int(words[1]) < 50:
        age = 2
    elif int(words[1]) >= 50:
        age = 2

    gender = int(words[2])  # 性别
    education = int(words[3])  # 教育
    marriageStatus = int(words[4])  # 婚姻状态
    haveBaby = int(words[5])
    hometown = int(words[6])
    residence = int(words[7])  # 常驻地
    # print([age, gender, education, marriageStatus, haveBaby, hometown, residence])
    return [age, gender, education, marriageStatus, haveBaby, hometown, residence]
