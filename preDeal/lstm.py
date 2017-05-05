from keras import layers
from keras.models import Sequential
from get_data import *

# 构建模型
'''
这个模型需要在后续，针对每一个人来做
'''

model_name = 'lstm'
HIDDEN_SIZE = 64
BATCH_SIZE = 128

model = Sequential()
model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(input_dim,)))
model.add(layers.LSTM(HIDDEN_SIZE, return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(1)))
model.add(layers.Activation('sigmoid'))
model.compile(loss="mse",
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

for iteration in range(1, 300):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=5,
              validation_data=(x_val, y_val))

model.save('model_log/' + model_name + '.h5')
