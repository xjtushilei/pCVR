from keras import layers
from keras.models import Sequential
from get_data import *

# 构建模型

model_name = 'dense'

HIDDEN_SIZE = 64
BATCH_SIZE = 128

model = Sequential()
model.add(layers.Dense(HIDDEN_SIZE, input_shape=(input_dim,), activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
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
              epochs=2,
              validation_data=(x_val, y_val))

model.save('model_log/'+model_name + '.h5')
