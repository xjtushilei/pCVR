import platform
from utils import *

linux_root_path = '/home/script/data/pcvr/'

if 'Windows' in platform.system():
    train_path = 'some_train.csv'
    # train_path = 'D://data/train.csv'
    print("windows")
if 'Linux' in platform.system():
    train_path = linux_root_path + 'train.csv'
    print("linux")


x = []
y = []
print('开始读取数据...')

with open(train_path) as file:
    all_the_text = [L.rstrip('\n') for L in file]
    print("总长度是", len(all_the_text) - 1)
    # 跳过首行
    for line in all_the_text[1:]:
        words = line.split(",")
        # 各种维度的待训练数据

        # 点击广告的时间
        clickTime = get_how_much_time(words[1])

        # 处理creativeID相关数据
        creativeID = int(words[3])


        userID = int(words[4])



        positionID = int(words[5])
        connectionType = int(words[6])
        telecomsOperator = int(words[7])

        temp = [clickTime, creativeID, userID, positionID, connectionType, telecomsOperator]
        x.append(temp)
        input_dim = len(temp)
        # 构建正确标签
        if words[0] == '1':
            exp = 1
        else:
            exp = 0
        y.append(exp)

print('问题个数是:', len(x))

# 前90%是训练数据，后10%是测试数据,通过反向传播来进行预测！
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

# print(x_val)
# print(y_val)