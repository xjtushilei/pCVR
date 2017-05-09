from keras import layers
from keras.models import Sequential
from get_data_v1 import *

# 构建模型

model_name = 'dense'

HIDDEN_SIZE = 64
BATCH_SIZE = 128

model = Sequential()
model.add(layers.Dense(HIDDEN_SIZE, input_shape=(1, 6)))
model.add(layers.Flatten())
# model.add(layers.TimeDistributed(layers.Dense(HIDDEN_SIZE)))
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

model.save('model_history_' + model_name + '.h5')
