import platform

import pickle

if 'Windows' in platform.system():
    root_path = 'D://data/'
    print("windows")
if 'Linux' in platform.system():
    root_path = '/home/script/data/pcvr/'
    print("linux")

user_app_actions_path = root_path + 'user_app_actions.csv'
user_installedapps_path = root_path + 'user_installedapps.csv'

with open(user_app_actions_path) as user_app_actions_file, open(user_installedapps_path) as user_installedapps_file:
    user_app_dict = {}

    user_app_actions_list = [L.rstrip('\n') for L in user_app_actions_file][1:]
    for line in user_app_actions_list:
        words = line.split(",")
        user_set = user_app_dict.get(int(words[0]), set())
        user_set.add(int(words[2]))
        user_app_dict[int(words[0])] = user_set
    print('字典大小', len(user_app_dict))

    user_installedapps_list = [L.rstrip('\n') for L in user_installedapps_file][1:]
    for line in user_installedapps_list:
        words = line.split(",")
        user_set = user_app_dict.get(int(words[0]), set())
        user_set.add(int(words[1]))
        user_app_dict[int(words[0])] = user_set
    print('最终字典大小', len(user_app_dict))
with open('user_app_dic.data', 'wb') as train_data_file:
    pickle.dump(user_app_dict, train_data_file)