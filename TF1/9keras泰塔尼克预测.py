# -*- coding: utf-8 -*-
# @Time    : 2020/1/18 8:41
# @Author  : 欧阳煜
# @Email   : 2455356027@qq.com


import urllib.request
import os

data_url='http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls'

data_file_dir='data/titanic3.xls'

if not os.path.exists(data_file_dir):
    result=urllib.request.urlretrieve(data_url,data_file_dir)
    print('data downloaded')
else:
    print(data_file_dir,'exist')

"""
pandas data preproduce
"""

import numpy as np
import pandas

df_data=pandas.read_excel('data/titanic3.xls')

selected_cols=['survived','name','pclass','sex','age','sibsp','parch','fare','embarked']
df_data=df_data[selected_cols]

#drop name
df_no_name=df_data.drop(['name'],axis=1)

# print(df_no_name.isnull().any()) # see those cols that may have null values
#cheak null
df_no_name['age']=df_no_name['age'].fillna(df_no_name['age'].mean())
df_no_name['fare']=df_no_name['fare'].fillna(df_no_name['fare'].mean())
df_no_name['embarked']=df_no_name['embarked'].fillna('S')
#sex code
df_no_name['sex']=df_no_name['sex'].map({'female':0,"male":1}).astype(int)
#enbarked code
df_no_name['embarked']=df_no_name['embarked'].map({'C':0,'Q':1,"S":2}).astype(int)

"""
   survived  pclass  sex      age  sibsp  parch      fare  embarked
0         1       1    0  29.0000      0      0  211.3375         2
1         1       1    1   0.9167      1      2  151.5500         2
2         0       1    0   2.0000      1      2  151.5500         2

"""

shuffled_data=df_no_name.sample(frac=1)

ndarray_data=shuffled_data.values
feature=ndarray_data[:,1:]#last 7 col
label=ndarray_data[:,0]

#normolization
from sklearn import preprocessing
norm_feature=preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(feature)

train_size=int(len(feature)*0.8)
x_train=feature[:train_size]
y_train=label[:train_size]

x_test=feature[train_size:]
y_test=label[train_size:]



"""
build model
"""
import tensorflow as tf
model=tf.keras.models.Sequential()

#add first layer
model.add(tf.keras.layers.Dense(units=64,
                                input_dim=7,#or input_shape=(7,)
                                use_bias=True,
                                kernel_initializer='uniform',
                                bias_initializer='zeros',
                                activation='relu'))

model.add(tf.keras.layers.Dropout(0.3))#dropout 防止过拟合

model.add(tf.keras.layers.Dense(units=32,activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# model.summary()   #

model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['acc'])

tensorboard_logdir="./logs"
ckpt_path='./ckpt/titanic.{epoch:02d}-{val_loss:.2f}.ckpt'

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir,histogram_freq=2),
    tf.keras.callbacks.ModelCheckpoint(file_path=ckpt_path,verbose=1,period=2)
]



train_history=model.fit(x=x_train,
                        y=y_train,
                        validation_split=0.2,
                        epochs=100,
                        callbacks=callbacks,
                        batch_size=40,
                        verbose=1)


import matplotlib.pyplot as plt
def visu_train_history(train_history,train_metric,validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title("Train History")
    plt.xlabel('epoch')
    plt.ylabel(train_metric)
    plt.legend(['train','validation'],loc='upper left')
    plt.show()

visu_train_history(train_history,'acc','val_acc')
visu_train_history(train_history,'loss','val_loss')
"""
模型评估
"""
evaluate_result=model.evaluate(x=x_test,y=y_test)
print(evaluate_result)

print(model.metrics_names)
"""
[0.5043336043831046, 0.8167939]
['loss', 'acc']
"""
surv_probility=model.predict(x_test)
print(surv_probility[:5])

