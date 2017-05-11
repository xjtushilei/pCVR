import pickle

import numpy
import tensorflow
from keras.models import load_model
from keras import layers
from keras.models import Sequential
from keras.utils import np_utils

from utils import *
from keras import backend as K

K.clear_session()

# 加载数据
print("开始反序列化加载数据...")
with open('get_data_v2.data', 'rb') as train_data_file:
    data = pickle.loads(train_data_file.read())
print("加载结束...")

(x, y, test_id, test) = data
x_test=x[0:5]
print(x_test)
x_test = np_utils.to_categorical(x_test)
print(len(x_test[0]))
model = load_model('lstm_v2.h5')
# model.save('lstm_v2.h5')
print("开始预测模型...")

# predict = model.predict(test[0:])
# print(predict)
# for i in test[0:10]:
#     print(test[0])



'''
print("开始将预测结果写入csv...")
with open('submission.csv', 'w') as file:
    file.write('instanceID,prob\n')
    index = 0
    for one in predict[:, 1:]:
        file.write(str(test_id[index]) + ',' + str(one[0]) + '\n')
        print(str(test_id[index]) + ',' + str(one[0]) + '\n',end='')
        index += 1

print("结束...")
'''