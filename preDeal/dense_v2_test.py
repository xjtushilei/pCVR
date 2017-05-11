import pickle

import numpy
import tensorflow
from keras import layers
from keras.models import Sequential
from utils import *
from keras import backend as K

K.clear_session()

# 加载数据
print("开始反序列化加载数据...")
with open('get_data_v2.data', 'rb') as train_data_file:
    data = pickle.loads(train_data_file.read())
print("加载结束...")

(x, y, test_id, test) = data
# 前90%是训练数据，后10%是测试数据,通过反向传播来进行预测！
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]
sample_weight = []
x_train = x_train[:, 0]
x_val = x_val[:, 0]
test = test[:, 0]

from keras.models import load_model

model = load_model('dense_v2.h5')
print("开始预测模型...")
predict = model.predict(test, verbose=1)
print(predict)
print("开始将预测结果写入csv...")
with open('dense_submission.csv', 'w') as file:
    file.write('instanceID,prob\n')
    index = 0
    for one in predict[:, 1:]:
        file.write(str(test_id[index]) + ',' + str(one[0]) + '\n')
        index += 1

print("结束...")
