import pickle
import platform
import numpy as np

from keras.utils import np_utils

from utils import *

if 'Windows' in platform.system():
    root_path = 'D://data/'
    print("windows")
if 'Linux' in platform.system():
    root_path = '/home/script/data/pcvr/'
    print("linux")

train_path = root_path + 'train.csv'
ad_path = root_path + 'ad.csv'
app_categories_path = root_path + 'app_categories.csv'
position_path = root_path + 'position.csv'
test_path = root_path + 'test.csv'
user_path = root_path + 'user.csv'
user_app_actions_path = root_path + 'user_app_actions.csv'
user_installedapps_path = root_path + 'user_installedapps.csv'

# train_path = 'some_train.csv'

x = []
y = []
test = []
test_id = []
print('开始读取数据...')

with open(train_path) as file, open(user_path) as user_file, \
        open(position_path) as position_file, open(app_categories_path) as app_categories_file, \
        open(ad_path) as ad_file, open(test_path) as test_file:
    all_the_text = [L.rstrip('\n') for L in file]
    print("训练总长度是", len(all_the_text) - 1)
    test_all_the_text = [L.rstrip('\n') for L in test_file]
    print("测试总长度是", len(test_all_the_text) - 1)

    print("开始生成app_categories字典...")
    app_categories_list = [L.rstrip('\n') for L in app_categories_file][1:]
    app_categories_dict = {}
    for line in app_categories_list:
        words = line.split(",")
        app_categories_dict[int(words[0])] = words[1]
    print("开始生成ad字典...")
    ad_list = [L.rstrip('\n') for L in ad_file][1:]
    ad_table_dict = {}
    for line in ad_list:
        words = line.split(",")
        ad_table_dict[int(words[0])] = line
    print("开始生成user字典...")
    user_list = [L.rstrip('\n') for L in user_file][1:]
    user_dict = {}
    for line in user_list:
        words = line.split(",")
        user_dict[int(words[0])] = line
    print("开始生成position字典...")
    position_list = [L.rstrip('\n') for L in position_file][1:]
    position_dict = {}
    for line in position_list:
        words = line.split(",")
        position_dict[int(words[0])] = line
    print('开始获得hot_dict')
    # creativeID_count_dict = {}
    # adID_dict_count_dict = {}
    # campaignID_count_dict = {}
    # advertiserID_count_dict = {}
    # app_categories_count_dict_big = {}
    # app_categories_count_dict_small = {}
    # connectionType_count_dict = {}
    # telecomsOperator_count_dict = {}
    # positionType_count_dict = {}
    # sitesetID_count_dict = {}
    # positionID_count_dict = {}
    #     for line in all_the_text[1:]:
    #         words = line.split(",")
    #         if words[0] == '1':
    #             exp = 1
    #         else:
    #             exp = 0
    #
    #         # 广告位ID：属于广告上下文信息
    #         positionID = int(words[5])
    #         positionID_count_dict_exp = positionID_count_dict.get(exp, {})
    #         positionID_count_dict_exp[positionID] = positionID_count_dict_exp.get(positionID, 0) + 1
    #         positionID_count_dict[exp] = positionID_count_dict_exp
    #         # 广告位类型：属于广告上下文信息
    #         positionType = int(position_dict[positionID].split(',')[2])
    #         positionType_count_dict_exp = positionType_count_dict.get(exp, {})
    #         positionType_count_dict_exp[positionType] = positionType_count_dict_exp.get(positionType, 0) + 1
    #         positionType_count_dict[exp] = positionType_count_dict_exp
    #         # 站点集合ID：属于广告上下文信息
    #         sitesetID = int(position_dict[positionID].split(',')[1])
    #         sitesetID_count_dict_exp = sitesetID_count_dict.get(exp, {})
    #         sitesetID_count_dict_exp[sitesetID] = sitesetID_count_dict_exp.get(sitesetID, 0) + 1
    #         sitesetID_count_dict[exp] = sitesetID_count_dict_exp
    #
    #         # 联网方式：属于广告上下文信息
    #         connectionType = int(words[6])
    #         connectionType_count_dict_exp = connectionType_count_dict.get(exp, {})
    #         connectionType_count_dict_exp[connectionType] = connectionType_count_dict_exp.get(connectionType, 0) + 1
    #         connectionType_count_dict[exp] = connectionType_count_dict_exp
    #
    #         # 运营商：属于广告上下文信息
    #         telecomsOperator = int(words[7])
    #         telecomsOperator_count_dict_exp = telecomsOperator_count_dict.get(exp, {})
    #         telecomsOperator_count_dict_exp[telecomsOperator] = telecomsOperator_count_dict_exp.get(telecomsOperator, 0) + 1
    #         telecomsOperator_count_dict[exp] = telecomsOperator_count_dict_exp
    #         # 素材
    #         creativeID = int(words[3])
    #         creativeID_count_dict_exp = creativeID_count_dict.get(exp, {})
    #         creativeID_count_dict_exp[creativeID] = creativeID_count_dict_exp.get(creativeID, 0) + 1
    #         creativeID_count_dict[exp] = creativeID_count_dict_exp
    #         # 账户
    #         advertiserID = int(ad_table_dict[creativeID].split(',')[3])
    #         advertiserID_count_dict_exp = advertiserID_count_dict.get(exp, {})
    #         advertiserID_count_dict_exp[advertiserID] = advertiserID_count_dict_exp.get(advertiserID, 0) + 1
    #         advertiserID_count_dict[exp] = advertiserID_count_dict_exp
    #         # 推广计划
    #         campaignID = int(ad_table_dict[creativeID].split(',')[2])
    #         campaignID_count_dict_exp = campaignID_count_dict.get(exp, {})
    #         campaignID_count_dict_exp[campaignID] = campaignID_count_dict_exp.get(campaignID, 0) + 1
    #         campaignID_count_dict[exp] = campaignID_count_dict_exp
    #         # 广告
    #         adID = int(ad_table_dict[creativeID].split(',')[1])
    #         adID_dict_count_dict_exp = adID_dict_count_dict.get(exp, {})
    #         adID_dict_count_dict_exp[adID] = adID_dict_count_dict_exp.get(adID, 0) + 1
    #         adID_dict_count_dict[exp] = adID_dict_count_dict_exp
    #         # 类别
    #         appID = int(ad_table_dict[creativeID].split(',')[4])
    #         app_categories = get_app_categories(app_categories_dict, appID)  # 广告类别
    #
    #         category = app_categories[0]  # 大类别
    #         app_categories_count_dict_exp = app_categories_count_dict_big.get(exp, {})
    #         app_categories_count_dict_exp[category] = app_categories_count_dict_exp.get(category, 0) + 1
    #         app_categories_count_dict_big[exp] = app_categories_count_dict_exp
    #         category = app_categories[1]  # 小类别
    #         app_categories_count_dict_exp = app_categories_count_dict_small.get(exp, {})
    #         app_categories_count_dict_exp[category] = app_categories_count_dict_exp.get(category, 0) + 1
    #         app_categories_count_dict_small[exp] = app_categories_count_dict_exp
    # count_dic = (creativeID_count_dict, adID_dict_count_dict, campaignID_count_dict, advertiserID_count_dict,
    #              app_categories_count_dict_big, app_categories_count_dict_small, connectionType_count_dict,
    #              telecomsOperator_count_dict, positionType_count_dict, sitesetID_count_dict, positionID_count_dict)
    # with open('count_dic.data', 'wb') as train_data_file:
    #     pickle.dump(count_dic, train_data_file)
    print("开始反序列化加载count_dic数据...")
    with open('count_dic.data', 'rb') as count_dic_file:
        count_dic = pickle.loads(count_dic_file.read())
    print("加载结束...")
    (creativeID_count_dict, adID_dict_count_dict, campaignID_count_dict, advertiserID_count_dict,
     app_categories_count_dict_big, app_categories_count_dict_small, connectionType_count_dict,
     telecomsOperator_count_dict, positionType_count_dict, sitesetID_count_dict, positionID_count_dict) = count_dic
    print("开始处理训练数据...")
    # 处理训练数据！跳过首行
    for line in all_the_text[1:]:
        words = line.split(",")
        # 各种维度的待训练数据
        '''
        构建正确标签
        '''
        if words[0] == '1':
            exp = 1
        else:
            exp = 0
        '''
        点击广告的时间
        '''
        clickTime = get_how_much_time_of_days(words[1])
        if clickTime >= 25 and exp == 0:
            continue
        '''
        处理creativeID相关数据
        '''
        creativeID = int(words[3])  # 素材id
        adID = int(ad_table_dict[creativeID].split(',')[1])  # 广告id
        campaignID = int(ad_table_dict[creativeID].split(',')[2])  # 推广计划id
        advertiserID = int(ad_table_dict[creativeID].split(',')[3])  # 账户id
        # 获取统计数据和所在区域的label
        app_categories_and_appPlatform = get_ad_info(ad_table_dict, app_categories_dict, creativeID)
        creativeID_float = get_percent(creativeID_count_dict, creativeID)
        creativeID_label = get_creativeID_num_label(get_label_1(creativeID_count_dict, creativeID))
        adID_float = get_percent(adID_dict_count_dict, adID)
        adID_label = get_adID_num_label(get_label_1(adID_dict_count_dict, adID))
        campaignID_float = get_percent(campaignID_count_dict, campaignID)
        campaignID_label = get_camgaignID_num_label(get_label_1(campaignID_count_dict, campaignID))
        advertiserID_float = get_percent(advertiserID_count_dict, advertiserID)
        advertiserID_label = get_advertiserID_num_label(get_label_1(advertiserID_count_dict, advertiserID))
        categories_big_float = get_percent(app_categories_count_dict_big, app_categories_and_appPlatform[0])
        categories_big_label = get_app_categories_count_dict_big_label(
            get_label_1(app_categories_count_dict_big, app_categories_and_appPlatform[0]))
        categories_small_float = get_percent(app_categories_count_dict_small, app_categories_and_appPlatform[1])
        categories_small_label = get_app_categories_count_dict_small_label(
            get_label_1(app_categories_count_dict_small, app_categories_and_appPlatform[1]))

        creativeID_info = app_categories_and_appPlatform + [creativeID_label,  adID_label,
                                                             campaignID_label,
                                                            advertiserID_label,
                                                             categories_big_label,
                                                             categories_small_label]
        '''
        处理user相关信息
        '''
        userID = int(words[4])
        userID_info = get_user_info(user_dict, userID)
        '''
        处理广告上下文信息
        '''
        # 广告位id
        positionID = int(words[5])
        positionID_float = get_percent(positionID_count_dict, positionID)
        positionID_label = get_positionID_num_label(get_label_1(positionID_count_dict, positionID))
        # 站点集合ID
        sitesetID = int(position_dict[positionID].split(',')[1])
        # 广告位类型
        positionType = int(position_dict[positionID].split(',')[2])
        # 联网方式：属于广告上下文信息
        connectionType = int(words[6])
        # 运营商：属于广告上下文信息
        telecomsOperator = int(words[7])
        positionID_info = [ positionID_label, sitesetID,  positionType,
                            connectionType,  telecomsOperator]


        temp = [clickTime] + creativeID_info + userID_info + positionID_info
        x.append(temp)
        y.append(exp)

    print("开始处理test数据...")
    # 处理test数据 , 跳过首行
    for line in test_all_the_text[1:]:
        words = line.split(",")
        # 测试数据的id
        instanceID = int(words[0])
        test_id.append(instanceID)

        '''
             点击广告的时间
        '''
        clickTime = get_how_much_time_of_days(words[2])
        '''
        处理creativeID相关数据
        '''
        creativeID = int(words[3])  # 素材id
        adID = int(ad_table_dict[creativeID].split(',')[1])  # 广告id
        campaignID = int(ad_table_dict[creativeID].split(',')[2])  # 推广计划id
        advertiserID = int(ad_table_dict[creativeID].split(',')[3])  # 账户id
        # 获取统计数据和所在区域的label
        app_categories_and_appPlatform = get_ad_info(ad_table_dict, app_categories_dict, creativeID)
        creativeID_float = get_percent(creativeID_count_dict, creativeID)
        creativeID_label = get_creativeID_num_label(get_label_1(creativeID_count_dict, creativeID))
        adID_float = get_percent(adID_dict_count_dict, adID)
        adID_label = get_adID_num_label(get_label_1(adID_dict_count_dict, adID))
        campaignID_float = get_percent(campaignID_count_dict, campaignID)
        campaignID_label = get_camgaignID_num_label(get_label_1(campaignID_count_dict, campaignID))
        advertiserID_float = get_percent(advertiserID_count_dict, advertiserID)
        advertiserID_label = get_advertiserID_num_label(get_label_1(advertiserID_count_dict, advertiserID))
        categories_big_float = get_percent(app_categories_count_dict_big, app_categories_and_appPlatform[0])
        categories_big_label = get_app_categories_count_dict_big_label(
            get_label_1(app_categories_count_dict_big, app_categories_and_appPlatform[0]))
        categories_small_float = get_percent(app_categories_count_dict_small, app_categories_and_appPlatform[1])
        categories_small_label = get_app_categories_count_dict_small_label(
            get_label_1(app_categories_count_dict_small, app_categories_and_appPlatform[1]))

        creativeID_info = app_categories_and_appPlatform + [ creativeID_label,  adID_label,
                                                             campaignID_label,
                                                            advertiserID_label,
                                                             categories_big_label,
                                                             categories_small_label]
        '''
        处理user相关信息
        '''
        userID = int(words[4])
        userID_info = get_user_info(user_dict, userID)
        '''
        处理广告上下文信息
        '''
        # 广告位id
        positionID = int(words[5])
        positionID_float = get_percent(positionID_count_dict, positionID)
        positionID_label = get_positionID_num_label(get_label_1(positionID_count_dict, positionID))
        # 站点集合ID
        sitesetID = int(position_dict[positionID].split(',')[1])
        sitesetID_float = get_percent(sitesetID_count_dict, sitesetID)
        # 广告位类型
        positionType = int(position_dict[positionID].split(',')[2])
        positionType_float = get_percent(positionType_count_dict, positionType)
        # 联网方式：属于广告上下文信息
        connectionType = int(words[6])
        connectionType_big_float = get_percent(connectionType_count_dict, connectionType)
        # 运营商：属于广告上下文信息
        telecomsOperator = int(words[7])
        telecomsOperator_big_float = get_percent(telecomsOperator_count_dict, telecomsOperator)
        positionID_info = [ positionID_label, sitesetID,  positionType,
                            connectionType,  telecomsOperator,
                           ]

        temp = [clickTime] + creativeID_info + userID_info + positionID_info
        test.append(temp)

print('训练集问题个数是:', len(x))
print('训练集标签个数是:', len(y))
print('test集问题个数是:', len(test))
print('test_id集问题个数是:', len(test_id))

# 正确的标签集合转化为noe-hot形式
# y = np_utils.to_categorical(y)

# print(y)
y = np.array(y)
print("y的shape:", y.shape)
test_id = np.array(test_id)
print("test_id的shape:", test_id.shape)
test = np.array(test)
print("test的shape:", test.shape)

x = np.array(x)
print("x的shape:", x.shape)

data = (x, y, test_id, test)
print("开始将所有数据序列化到本地...")
with open('get_data_v4.3.data', 'wb') as train_data_file:
    pickle.dump(data, train_data_file)
# 前90%是训练数据，后10%是测试数据,通过反向传播来进行预测！
# split_at = len(x) - len(x) // 10
# (x_train, x_val) = x[:split_at], x[split_at:]
# (y_train, y_val) = y[:split_at], y[split_at:]
#
# # print(x_val)
# print(x_val.shape)
# print(y_val.shape)
# print(y_val)
print("结束...")
