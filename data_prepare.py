#!usr/bin/env python
# -*- coding: utf-8 -*-
# 此代码的功能只是一个格式转换的功能，将原始数据（PRSA_data_2010.1.1-2014.12.31.csv里面的文件）进行处理，
# 然后处理好的数据存到pollution.csv文件。
"""
Time:18/1/9
---------------------------
Question:   时间序列问题，利用前几天的空气污染数据预测下一段时间的空气污染情况
            Basic Data Preparation（存在的问题：2000多条数据中 多条数据 pm2.5 为空值NA --> 补为0）
---------------------------
"""

import pandas as pd
from datetime import datetime
from model.util import RAW_DATA, PROCESS_LEVEL1

pd.options.display.expand_frame_repr = False

# raw_data = pd.read_csv(RAW_DATA)
# print(raw_data.head())

# 处理时间，字符串 ---> 时间格式
def parsedate(x):
    return datetime.strptime(x, '%Y %m %d %H')


# index_col: 指定索引列。
# 关注对时间处理的模块
raw_data = pd.read_csv(RAW_DATA, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parsedate)
# 去掉原始数据的No列
raw_data.drop('No', axis=1, inplace=True)
# 指定列名
raw_data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
# 加入新的表头date
raw_data.index.name = 'date'

raw_data['pollution'].fillna(0, inplace=True)
# 只选取了raw_data从24行开始的数据，也就是说删除了24行之前的数据
raw_data = raw_data[24:]   
# print(raw_data.head())
# 讲处理好的数据存到PROCESS_LEVEL1（PROCESS_LEVEL1 = 'resource/pollution.csv'）文件下
raw_data.to_csv(PROCESS_LEVEL1)
