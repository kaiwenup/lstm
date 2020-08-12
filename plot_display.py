#!usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
from model.util import  RAW_DATA_2

pd.options.display.expand_frame_repr = False

column_names = ['user-id',
                'activity',   # 标签
                'timestamp',  # 时间戳
                'co-fli',     # 以下三个表头为卡尔曼滤波后的数据
                'smog-fli',
                't-fli'
                ]
dataset = pd.read_csv(RAW_DATA_2,
                     header=None, # 指定行数用来作为列名，数据开始行数。如果文件中没有列名，则默认为0，否则设置为None。
                     names=column_names)


# loads the 'pollution.csv' and treat each column as a separate subplot
values = dataset.values
#groups = [0, 1, 2, 3, 4, 5]
groups = [3, 4, 5]

i = 3
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()

# print(dataset.head())
