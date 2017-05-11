import pickle

import numpy
import tensorflow
from keras import layers, optimizers
from keras.models import Sequential
from utils import *
from keras import backend as K

K.clear_session()

# 加载数据
print("开始反序列化加载数据...")
with open('get_data_v3.data', 'rb') as train_data_file:
    data = pickle.loads(train_data_file.read())
print("加载结束...")

(x, y, test_id, test) = data
print(test_id[2])
# 前90%是训练数据，后10%是测试数据,通过反向传播来进行预测！
split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

# print(x_val)
print('x-shape：', x_val.shape)
print('y-shape：', y_val.shape)
print('test_id-shape：', test_id.shape)
print('test-shape：', test.shape)

print("设置显卡信息...")
# 设置tendorflow对显存使用按需增长
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

print("开始构建模型...")
# 构建模型
model_name = 'lstm'
HIDDEN_SIZE = 256
BATCH_SIZE = 5000
# LSTM = layers.LSTM
model = Sequential()
print('首层输入维度是:', x_val.shape[2])
# shape of (len_of_sequences, nb_of_features)
model.add(layers.normalization.BatchNormalization(input_shape=(1, x_val.shape[2])))
model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(1, x_val.shape[2]), return_sequences=True))
model.add(layers.normalization.BatchNormalization())
model.add(layers.LSTM(HIDDEN_SIZE // 2))
model.add(layers.Dense(HIDDEN_SIZE))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))
sgd = optimizers.SGD(lr=1, clipvalue=0.5)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', 'binary_accuracy'])

model.summary()
print("开始训练模型...")
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=200,
          validation_data=(x_val, y_val))

model.save('lstm_v3.h5')
print("开始预测模型...")
predict = model.predict(test[0:], batch_size=5000, verbose=1)
print(predict[:, 1:50])
print("开始将预测结果写入csv...")
with open('lstm_v3_submission.csv', 'w') as file:
    file.write('instanceID,prob\n')
    index = 0
    for one in predict[:, 1:]:
        file.write(str(test_id[index]) + ',' + str(one[0]) + '\n')
        index += 1

print("结束...")
