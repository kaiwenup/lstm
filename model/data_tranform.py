#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
---------------------------
Question: Series  --->  Supervised Learning Problem
    1、时间序列预测，以前一时刻（t-1）的所有数据预测当前时刻（t）的值

    X = PM2.5(t-1)  pollution(t-1) ,dew(t-1) ,temp(t-1) ,press(t-1) ,wnd_dir(t-1) ,wnd_spd(t-1) ,snow(t-1) ,rain(t-1)
    Y = PM2.5(t)

    2、在做inversed_transformed 时，需要注意的是所有的维度需要保持一致
---------------------------
"""
import pandas as pd
from model.util import PROCESS_LEVEL1
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from model.series_to_supervised_learning import series_to_supervised
pd.options.display.expand_frame_repr = False


dataset = pd.read_csv(PROCESS_LEVEL1, header=0, index_col=0)
# 读取表头
dataset_columns = dataset.columns
# 读取数据
values = dataset.values

# 打印数据的前24行
#print(dataset.values[0:24])

# 对第四列（风向）数据进行编码，也可进行 哑编码处理
# 因为风向数据是通过字母来表示的，所以需要进行编码
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])  # 只对第四行的数据进行编码，也就是wnd_dir
values = values.astype('float32')

# 对数据进行归一化处理, valeus.shape=(, 8),inversed_transform时也需要8列
scaler = MinMaxScaler(feature_range=(0, 1))
# scaled为归一化之后的数据
scaled = scaler.fit_transform(values)


# 将序列数据转化为监督学习数据
reframed = series_to_supervised(scaled, dataset_columns, 1, 1)
# 因为只是通过t-1时刻的数据预测t时刻的数据，所以删除了t时刻除pollution的数据，也就是9～15列的数据
reframed.drop(reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1, inplace=True)
# drop()函数：删除行和列
# drop([ ],axis=0,inplace=True)
# drop([])，默认情况下删除某一行，如果要删除某列，需要axis=1
# 参数inplace 默认情况下为False，表示保持原来的数据不变，True 则表示在原来的数据上改变。


# 将所有数据转化为训练集和测试集
# train：第一年的所有数据 test：除列第一年的其他数据
values = reframed.values
n_train_hours = 365 * 24
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# 监督学习结果划分,test_x.shape = (, 8)
# train[:, :-1]：取第一列到倒数第二列的数据
# train[:, -1]：取最后一列的数据（也就是pollution的数据）
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# 为了在LSTM中应用该数据，需要将其格式转化为3D format，即[Samples, timesteps, features]
# train_x.shape[0]：行数  train_x.shape[1]：列数
train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1])) 
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
# train_X.shape[0]:8760  train_X.shape[0]:1  train_X.shape[0]:8
