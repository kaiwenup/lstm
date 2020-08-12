# 基于Keras的LSTM多变量时间序列预测

短期记忆网络：Long Short-term memory (LSTM) Network

[本项目源代码](https://github.com/kaiwenup/lstm)

本程序基于GitHub开源代码修改（[代码链接](https://github.com/634671436/Air_Pollution_Forcast_Beijing)）

## 文件目录结构

```markdown
lstm
 -lstm.py
 -.gitignore
 -data_prepare.py
 -plot_display.py
 -README.md
 -model
  -data_tranform.py
  -series_to_supervised_learning.py
  -util.py
 -resource
  -fire_data.txt
```

`lstm.py`：主程序文件

`.gitignore`：git中设置无需纳入git管理的文件

`data_prepare.py`：数据预处理程序

`plot_display.py`：数据可视化程序

`README.md`：说明文件

`model`：存放子程序的文件，其中：

- `data_tranform.py`：数据转化子程序
- `series_to_supervised_learning.py`：监督学习子程序
- `util.py`：存放原始数据的存放路径，以及预处理后的数据存放的路径

`resource`：存放原始数据

- `fire_data.txt`：存放原始数据

<!--more-->

## 环境搭建

详见[基于Keras的一维卷积神经网络](http://39.106.110.164/2020/08/06/计算机-DeepLearning-基于Keras的一维卷积神经网络/)

搭建好环境之后，在lstm文件夹下运行依次运行`python3 data_prepare.py`和`python3 lstm.py`。没问题则环境搭建完成。

## LSTM数据处理

### 原始数据构成

所有数据都存放在txt文档中（也可以存在csv文件中），存放路径为`fire_data/fire_data_raw.txt`，每一行都以`;`结尾

第一列数据为数据所在的组别，在本数据集中分为两个组别，分别是1和2。其中1表示训练集的数据，2表示为测试集的数据。第二列为所在列的数据标签，标识着这一列数据对应的状态，本数据集中一共有两个标签`fire`和`nofire`。第三列表示时间戳。第四、五、六列分别表示采集到的一氧化碳、烟雾以及温度数据（所有数据都做了放大处理，放大倍数为100）。

### 数据预处理

首先介绍`model/util.py`。该文件主要存放需要导入和导出的数据文件的路径。代码如下：

```python
RAW_DATA_2 = 'resource/fire_data_raw.txt'
PROCESS_LEVEL2 = 'resource/fire_data.txt'
```

所以，`RAW_DATA_2`代表着需要输入的数据文件路径，`PROCESS_LEVEL2`代表着输出的数据文件的路径。

在运行`lstm.py`之前，需要运行`data_prepare.py`。该文件对原始数据文件进行预处理，然后将处理好的数据存放到`PROCESS_LEVEL2`路径下。代码如下。

```python
#表头的定义
column_names = ['user-id',
                'activity',   # 标签
                'timestamp',  # 时间戳
                'co-fli',     # 以下三个表头为卡尔曼滤波后的数据
                'smog-fli',
                't-fli'
                ]
#读取数据，然后将表头加入数据中
raw_data = pd.read_csv(RAW_DATA_2,
                     header=None,
                     names=column_names)
#去掉user-id，因为暂时用不到
raw_data.drop('user-id', axis=1, inplace=True)
#用正则表达式去掉 ；
raw_data['t-fli'].replace(regex=True,
    inplace=True,
    to_replace=r';',
    value=r'')
#因为将最后一列数据转化为浮点数
raw_data['t-fli'] = raw_data['t-fli'].apply(convert_to_float)
#导出数据
raw_data.to_csv(PROCESS_LEVEL2)

def convert_to_float(x):

    try:
        return np.float(x)
    except:
        return np.nan
```

#### 导入预处理后数据

以下代码在`model/data_tranform.py`

导入经过预处理好的数据，然后将表头存到`dataset_columns`，然后将其他数据存到`values`

```python
dataset = pd.read_csv(PROCESS_LEVEL2, header=0, index_col=0)
dataset_columns = dataset.columns
values = dataset.values
```

### 标签处理

标签也就是`activity`所在的那一列，是以字符串的形式出现的，需要转化为数字形式，实现代码：

```python
encoder = LabelEncoder()
values[:, 0] = encoder.fit_transform(values[:, 0])  # 只对第零行的数据进行编码，也就是activity
values = values.astype('float32')
```

### 数据归一化

代码实现如下：


```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
```

### 数据类型转换

keras只识别`float32`类型的数据，所以要进行类型转换。

代码实现如下：

```python
values = values.astype('float32')
```

### 序列数据转化为监督学习数据^*^

以下代码代码在`model/series_to_supervised_learning.py`

序列数据转化为监督学习数据的本质也是对数据进行转换和重组`re-frame`，以便能够将输入导入LSTM训练模型中

因为LSTM模型的的功能就是用以往的数据来预测将来某个时刻的数据，所以输入数据中的每一行也应该包含历史数据和需要预测的数据。在本程序中只是用t-1时刻的传感器数据和t时刻的传感器数据来预测t时刻的activity(fire/nofire)，所以数据的每一行应该包含以下数据：

- 上一刻传感器数据（t-1时刻传感器数据）
- 本时刻传感器数据（t时刻传感器数据）
- 本时刻的activity（fire/nofire）（t时刻activity）

数据构建过程如下：

![监督学习](https://i.loli.net/2020/08/10/bCHErwcRstF1uIQ.png)

图中Segment1和Segment2都是Segment的备份，数据完全一样。数据构建分三步进行：

第一步：Segment2不动，Segment1相对于Segment2向下移动一行，然后按行拼接两个数据。此时的数据一共有十列

第二步：第一列数据和最后一列数据不可用，所以舍弃

第三步：去掉第0，1，6列的元素，然后将第5列的元素移动到最后一列

实现代码如下：

```python
#model/series_to_supervised_learning.py
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

#model/data_tranform.py
df_id=reframed.ix[:,5]
reframed.drop(reframed.columns[[0, 1, 5, 6]], axis=1, inplace=True)
reframed.insert(6,'activity1(t)',df_id)
```

#### 参考博客：

- [numpy数组切片操作之[:,2]、[-1:,0:2]、[1:,-1:]等都是啥？](https://blog.csdn.net/qq_41375609/article/details/95027651)
- [numpy多维数组shape的理解](https://blog.csdn.net/u013894427/article/details/88894826)
- [python中 x[:,0]和x[:,1] 理解和实例解析](https://blog.csdn.net/u014159143/article/details/80307717)

- [Time Series Forecasting as Supervised Learning](https://machinelearningmastery.com/time-series-forecasting-supervised-learning/)
- [How to Convert a Time Series to a Supervised Learning Problem in Python](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

### 划分训练集和测试集

```python
values = reframed.values
n_train_hours =  30000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]
train_X = train_x.reshape((train_x.shape[0], 1, train_x.shape[1])) 
test_X = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
```

数据的前30000行为训练数据，30000行以后为测试数据。

## 搭建LSTM模型以及数据拟合

```python
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y))

```

LSTM模型中，隐藏层有50个神经元，损失函数采用Mean Absolute Error(MAE)，优化算法采用Adam，模型采用100个epochs并且每个batch的大小为72。最后，在fit()函数中设置validation_data参数，记录训练集和测试集的损失，并在完成训练和测试后绘制损失图。

## 模型评估

接下里我们对模型效果进行评估。

值得注意的是：需要将预测结果和部分测试集数据组合然后进行比例反转（invert the scaling），同时也需要将测试集上的预期值也进行比例转换。

**至于在这里为什么进行比例反转，是因为我们将原始数据进行了预处理（连同输出值y），此时的误差损失计算是在处理之后的数据上进行的，为了计算在原始比例上的误差需要将数据进行转化。（反转时的矩阵大小一定要和原来的大小（shape）完全相同，否则就会报错。**）

通过以上处理之后，再结合RMSE（均方根误差）计算损失

```python
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
```

### 参考博客：

- [基于Keras的LSTM多变量时间序列预测](https://blog.csdn.net/qq_28031525/article/details/79046718)
- [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
- [The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)