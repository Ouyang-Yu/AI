from tensorflow import keras
import pandas
import numpy

df=pandas.read_csv("x.csv", header=0,encoding='gbk')
print(df.shape)
x_train=numpy.array(df.values)

df1=pandas.read_csv("y.csv", header=0,encoding='gbk')
print(df1.shape)
y_train=numpy.array(df1.values)


for i in range(10):
    x_train[:,i]=(x_train[:,i]-x_train[i].min())/(x_train[:,i].max()-x_train[:,i].min())

model=keras.models.Sequential([
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(12,activation=keras.activations.sigmoid)
])
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss=keras.losses.mse,
    metrics=keras.metrics.Accuracy
)

model.fit(x_train,y_train,epochs=100,batch_size=100)
