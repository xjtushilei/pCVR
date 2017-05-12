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
    ad_dict = {}
    for line in ad_list:
        words = line.split(",")
        ad_dict[int(words[0])] = line
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

    print("开始处理训练数据...")
    # 处理训练数据！跳过首行
    for line in all_the_text[1:]:
        words = line.split(",")
        # 各种维度的待训练数据

        # 点击广告的时间
        clickTime = get_how_much_time_of_days(words[1])

        # 处理creativeID相关数据
        creativeID = int(words[3])
        creativeID_info = get_ad_info(ad_dict, app_categories_dict, creativeID)
        # 处理user相关信息
        userID = int(words[4])
        userID_info = get_user_info(user_dict, userID)

        # 处理 广告位置信息：属于广告上下文信息
        positionID = int(words[5])
        positionID_info = get_position_info(position_dict, positionID)

        # 联网方式：属于广告上下文信息
        connectionType = int(words[6])
        # 运营商：属于广告上下文信息
        telecomsOperator = int(words[7])

        temp = [clickTime] + creativeID_info + userID_info + positionID_info + [connectionType, telecomsOperator]
        temp_array_list = []
        for one_temp in temp:
            temp_array_list.append(one_temp)
        # print(temp_array_list)
        x.append(temp_array_list)
        # print(x)
        # 构建正确标签
        if words[0] == '1':
            exp = 1
        else:
            exp = 0
        y.append(exp)
    print("开始处理test数据...")
    # 处理test数据 , 跳过首行
    for line in test_all_the_text[1:]:
        words = line.split(",")
        # 测试数据的id
        instanceID = int(words[0])
        test_id.append(instanceID)

        # 点击广告的时间
        clickTime = get_how_much_time_of_days(words[2])
        # 处理creativeID相关数据
        creativeID = int(words[3])
        creativeID_info = get_ad_info(ad_dict, app_categories_dict, creativeID)
        # 处理user相关信息
        userID = int(words[4])
        userID_info = get_user_info(user_dict, userID)
        # 处理 广告位置信息：属于广告上下文信息
        positionID = int(words[5])
        positionID_info = get_position_info(position_dict, positionID)
        # 联网方式：属于广告上下文信息
        connectionType = int(words[6])
        # 运营商：属于广告上下文信息
        telecomsOperator = int(words[7])
        temp = [clickTime] + creativeID_info + userID_info + positionID_info + [connectionType, telecomsOperator]
        test.append([temp])


print('训练集问题个数是:', len(x))
print('标签个数是:', len(y))
print('test集问题个数是:', len(test))
print('test_id集问题个数是:', len(test_id))

# 正确的标签集合转化为noe-hot形式
y = np_utils.to_categorical(y)

# print(y)
x = np.array(x)
y = np.array(y)
print("x的shape:", x.shape)
test_id = np.array(test_id)
test = np.array(test)
data = (x, y, test_id, test)
print("开始将所有数据序列化到本地...")
with open('get_data_v4.data', 'wb') as train_data_file:
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