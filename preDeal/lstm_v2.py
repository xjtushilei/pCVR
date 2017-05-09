import pickle
import tensorflow
from keras import layers
from keras.models import Sequential

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
HIDDEN_SIZE = 128
BATCH_SIZE = 128
# LSTM = layers.LSTM
model = Sequential()
print('首层输入维度是:', x_val.shape[2])
# shape of (len_of_sequences, nb_of_features)
model.add(layers.LSTM(HIDDEN_SIZE, input_shape=(1, x_val.shape[2]), return_sequences=True))
model.add(layers.LSTM(HIDDEN_SIZE // 2))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))
model.compile(loss='msle',
              optimizer='adam',
              metrics=['accuracy', 'binary_accuracy', 'sparse_categorical_accuracy'])

model.summary()
print("开始训练模型...")
model.fit(x_train, y_train,
          batch_size=BATCH_SIZE,
          class_weight={1: 200, 0: 1},
          epochs=50,
          validation_data=(x_val, y_val))

print("开始预测模型...")
predict = model.predict_proba(test)

print("开始将预测结果写入csv...")
with open('submission.csv', 'w') as file:
    file.write('instanceID,prob')
    for index, one in enumerate(predict[:, 1:]):
        file.write(test_id[index] + ',' + str(one[0]) + '\n')

print("结束...")
