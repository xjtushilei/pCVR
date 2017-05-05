from keras import layers
from keras.models import Sequential
from get_data import *

# 构建模型

model_name = 'pnn'

HIDDEN_SIZE = 64
BATCH_SIZE = 128

model = Sequential()
print('input_dim :', input_dim)
model.add(layers.Embedding(input_dim, HIDDEN_SIZE))
model.add(layers.Dense(64))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))
model.compile(loss="msle",
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

model.save('model_log/' + model_name + '.h5')
