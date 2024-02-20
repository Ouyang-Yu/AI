from datetime import date
import quandl
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
import tensorflow.keras
import numpy as np
start = date(2000,10,12)
end = date.today()
google_stock = pd.DataFrame(quandl.get("WIKI/GOOGL", start_date=start, end_date=end))
print(google_stock.shape)
google_stock.tail()
google_stock.head()

plt.figure(figsize=(16, 8))
plt.plot(google_stock['Close'])
plt.show()


# 时间点长度
time_stamp = 50


# 划分训练集与验证集
google_stock = google_stock[['Open', 'High', 'Low', 'Close', 'Volume']]  #  'Volume'
train = google_stock[0:2800 + time_stamp]
valid = google_stock[2800 - time_stamp:]


# 归一化
scaler = tensorflow.keras.utils.MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_tra.form(train)
x_train, y_train = [], []




# 训练集
print(scaled_data.shape)
print(scaled_data[1, 3])
for i in range(time_stamp, len(train)):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i, 3])


x_train, y_train = np.array(x_train), np.array(y_train)


# 验证集
scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid)):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i, 3])


x_valid, y_valid = np.array(x_valid), np.array(y_valid)


print(x_train.shape)
print(x_valid.shape)
train.head()
