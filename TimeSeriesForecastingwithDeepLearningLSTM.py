import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV filcondae I/O (e.g. pd.read_csv)
import random as rd  # generating random numbers
import matplotlib.pyplot as plt  # basic plotting
import seaborn as sns  # for prettier plots
import plotly.express as px

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Setup notebook
from pathlib import Path

# import necessary package
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import TimeseriesGenerator

import sklearn
from sklearn import preprocessing

# 畫圖表用
import matplotlib.pyplot as plt

from pandasql import sqldf
dfsql = lambda q: sqldf(q, globals())

tf.debugging.set_log_device_placement(True)


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

for dirname, _, filenames in os.walk('d:/lge/pycharm-projects/kaggle_store_sales/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# if you are interesting in faster version(~minutes) but lower performance
# you can check ver.5/6 of this notebook
# it might help

# read the data
comp_dir = Path('d:/lge/pycharm-projects/kaggle_store_sales/input/')

store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    usecols=['store_nbr', 'family', 'date', 'sales', 'onpromotion'],
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'sales': 'float32',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)

test = pd.read_csv(
    comp_dir / 'test.csv',
    dtype={
        'store_nbr': 'category',
        'family': 'category',
        'onpromotion': 'uint32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
)

oil = pd.read_csv(
    comp_dir / 'oil.csv',
    parse_dates=['date'],
    infer_datetime_format=True,
)

# fill missing date
oil = oil.set_index("date").asfreq(freq="D")

# fill the NaN value by interpolation
oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(limit_direction="both")

store_sales = store_sales.merge(oil, on="date")
test = test.merge(oil, on="date")

dfsql(f'''
--SELECT substr(date,1,10) 
SELECT count(*)
FROM oil
''').head(3)


# %%

def series_to_supervised(data, n_in=1, n_out=1, futureArr=None, targetCol=None, dropnan=True):
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
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s(t-%d)' % (j, i)) for j in df.columns]

    # forecast sequence (t, t+1, ... t+n)
    if futureArr != None:
        for i in range(0, n_out):
            for futureCol in futureArr:
                cols.append(df.shift(-i)[futureCol])
                if i == 0:
                    names += [('%s(t)' % (futureCol))]
                else:
                    names += [('%s(t+%d)' % (futureCol, i))]

    for i in range(0, n_out):
        if targetCol == None:
            cols.append(df.shift(-i))
            if i == 0:
                names += [('%s(t)' % (j)) for j in df.columns]
            else:
                names += [('%s(t+%d)' % (j, i)) for j in df.columns]
        else:
            cols.append(df.shift(-i)[targetCol])
            if i == 0:
                names += [('%s(t)' % (targetCol))]
            else:
                names += [('%s(t+%d)' % (targetCol, i))]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


store_nbr_types = store_sales["store_nbr"].unique()

family_types = store_sales["family"].unique()


for store_nbr_type in store_nbr_types:
    for family_type in family_types:
        train_data = store_sales[(store_sales["store_nbr"] == store_nbr_type) & (store_sales["family"] == family_type)]

        train_data = train_data.reset_index()
        train_data = train_data.drop(columns=["index", "date", "store_nbr", "family"])

        test_data = test[(test["store_nbr"] == store_nbr_type) & (test["family"] == family_type)]
        test_data = test_data.drop(columns=["date", "store_nbr", "family"])
        break
    break

# concat train and test data
total_data = pd.concat([train_data, test_data]).drop(columns=["id"])

# Normalization
feature_name = total_data.columns

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

total_data = scaler.fit_transform(total_data)

total_data = pd.DataFrame(total_data, columns=feature_name)

# example
# 使用5天的資料，來預測後面兩天的資料
futureArr = ["onpromotion", "dcoilwtico"]

series_to_supervised(total_data, 5, 2, futureArr=futureArr, targetCol="sales")

# 要用幾天來預測
past_days = 50
# 預測未來幾天
predict_days = 16
# 未來資料要使用於input的欄位
futureArr = ["onpromotion", "dcoilwtico"]
# 所要預測的欄位
targetCol = "sales"

train = series_to_supervised(total_data, past_days, predict_days, futureArr, targetCol)

split_ratio = 0.8

split_number = np.floor(len(train.index) * split_ratio)
split_number = np.int(split_number)

values = train.values

# split into train and validation sets
train = values[:split_number, :]
val = values[split_number:, :]

# split into input and outputs
train_x, train_y = train[:, :-predict_days], train[:, -predict_days:]
val_x, val_y = val[:, :-predict_days], val[:, -predict_days:]
# reshape input to be 3D [samples, timesteps, features]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
val_x = val_x.reshape((val_x.shape[0], 1, val_x.shape[1]))

# 預備用於預測的data(-17:-16代表使用倒數第17天的資料(2017/08/15))
prediction_data = series_to_supervised(total_data, past_days, predict_days, futureArr, targetCol, dropnan=False).values[-17:-16, :-predict_days]
prediction_data = prediction_data.reshape((prediction_data.shape[0], 1, prediction_data.shape[1]))

print(train_x.shape, train_y.shape, val_x.shape, val_y.shape, prediction_data.shape)
#%%
# Model
model = keras.models.Sequential([
    keras.layers.LSTM(units=30, return_sequences=True, input_shape=(train_x.shape[1], train_x.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(units=30, return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(predict_days))
])

optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])

model.summary()

early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

model_result = model.fit(train_x, train_y, epochs=100, batch_size=32, validation_data=(val_x, val_y), verbose=2, shuffle=False, callbacks=[early_stopping])

# plot history
plt.figure(figsize=(30, 10))

plt.subplot(1, 2, 1)
plt.plot(model_result.history["loss"], label="training")
plt.plot(model_result.history["val_loss"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model_result.history["mse"], label="training")
plt.plot(model_result.history["val_mse"], label="validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

prediction = model.predict(prediction_data)

prediction = np.squeeze(prediction) / scaler.scale_[0]

test_data["sales"] = prediction

test_data
