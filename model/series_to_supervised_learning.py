#!usr/bin/env python
# -*- coding: utf-8 -*-

"""
Time:17/1/1
---------------------------
Question: LSTM data Preparation(normalizing the input variables)
          How to Convert a Time Series to a Supervised Learning Problem in Python?

---------------------------
"""
import pandas as pd


def series_to_supervised(data, columns, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    # 如果传入的数据是list则n_vars为1,否则为传入数据的列数
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 构造(t-n, ... t-1)的数据
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s%d(t-%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    # 构造(t, t+1, ... t+n)的数据
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s%d(t)' % (columns[j], j + 1)) for j in range(n_vars)]
        else:
            names += [('%s%d(t+%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
    # put it all together
    # 对应行将数组进行拼接
    agg = pd.concat(cols, axis=1)
    # 加表头
    agg.columns = names
    # drop rows with NaN values
    # 去掉含Nan数据
    if dropnan:
        clean_agg = agg.dropna()
    return clean_agg
    # return agg


if __name__ == '__main__':
    values = [x for x in range(10)]
    data = series_to_supervised(values, ['temp'], 2)
    print(data)