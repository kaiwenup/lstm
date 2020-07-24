#!usr/bin/env python
# -*- coding: utf-8 -*-

from keras import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from numpy import concatenate  # 数组拼接
from math import sqrt
from sklearn.metrics import mean_squared_error

#自建函数库
from model.data_tranform import scaler, test_x, train_X, test_X, train_y, test_y

# LSTM模型中，隐藏层有50个神经元，输出层1个神经元（回归问题），输入变量是一个时间步（t-1）的特征，
# 损失函数采用Mean Absolute Error(MAE)，优化算法采用Adam，模型采用50个epochs并且每个batch的大小为72。
# 最后，在fit()函数中设置validation_data参数，记录训练集和测试集的损失，并在完成训练和测试后绘制损失图。

# create an instance of the Sequential class
model = Sequential()
# LSTM recurrent layer
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
# fully connected layer
model.add(Dense(1))
# optimizer:the optimization algorithm to use to train the network
# loss:the loss function used to evaluate the network that is minimized by the optimization algorithm
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y))
# 运行的时候不显示相关打印信息
#history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y),verbose=0)
# 数据可视化
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# make the prediction
# 为了在原始数据的维度上计算损失，需要将数据转化为原来的范围再计算损失
yHat = model.predict(test_X)

# 这里注意的是保持拼接后的数组的列数需要与之前的保持一致
inv_yHat = concatenate((yHat, test_x[:, 1:]), axis=1)   # 数组拼接, axis=1表示对应行将数组进行拼接
inv_yHat = scaler.inverse_transform(inv_yHat) # 将归一化的数据转化为原来的范围
inv_yHat = inv_yHat[:, 0]

# 将test_y重新变成一维数组
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)    
inv_y = inv_y[:, 0]

rmse = sqrt(mean_squared_error(inv_yHat, inv_y))
print('Test RMSE: %.3f' % rmse)