import tensorflow
from keras import layers
from keras.models import Sequential
from get_data_v1 import *

# 构建模型
'''
这个模型需要在后续，针对每一个人来做
'''


# 设置tendorflow对显存使用按需增长
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.Session(config=config)

model_name = 'lstm'
HIDDEN_SIZE = 64
BATCH_SIZE = 128
# LSTM = layers.LSTM
model = Sequential()
print('首层输入维度是:', input_dim)
# shape of (len_of_sequences, nb_of_features)
model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(1, 6), return_sequences=True))
model.add(layers.LSTM(HIDDEN_SIZE // 2))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))
model.compile(loss="mse",
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          class_weight={1: 100, 0: 1},
          epochs=50,
          validation_data=(x_val, y_val))
# model.save('model_history_' + model_name + '.h5')
