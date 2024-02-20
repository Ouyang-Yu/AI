import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df=pd.read_csv("boston.csv", header=0)#同一文件夹下
print(df.describe())
"""
             CRIM         ZN       INDUS   ...     PTRATIO       LSTAT        MEDV
count  506.000000  506.000000  506.000000  ...  506.000000  506.000000  506.000000
mean     3.613524   11.363636   11.136779  ...   18.455534   12.653063   22.532806
std      8.601545   23.322453    6.860353  ...    2.164946    7.141062    9.197104
min      0.006320    0.000000    0.460000  ...   12.600000    1.730000    5.000000
25%      0.082045    0.000000    5.190000  ...   17.400000    6.950000   17.025000
50%      0.256510    0.000000    9.690000  ...   19.050000   11.360000   21.200000
75%      3.677082   12.500000   18.100000  ...   20.200000   16.955000   25.000000
max     88.976200  100.000000   27.740000  ...   22.000000   37.970000   50.000000

[8 rows x 13 columns]
"""


df=np.array(df.values) #获取值转为二维数组


for i in range(12):
    df[:,i]=(df[:,i]-df[i].min())/(df[:,i].max()-df[:,i].min())
"""
对多变量数据做0-1归一化处理  10万 与500平方米能一起计算么   即(f-min)/(max-min)
"""

feature=df[:,:12]
label=df[:,12]#最后一列为结果   516行.1列的矩阵

x=tf.placeholder(tf.float32,[None,12]) #占位符,是一个12列的矩阵,行数未知
y=tf.placeholder(tf.float32,[None,1]) #占位符,是一个1列的矩阵,行数未知

#定义命名空间
with tf.name_scope("Model"):
    w=tf.Variable(tf.random_normal([12,1],stddev=0.01))  #权重为12行的矩阵
    b=tf.Variable(1.0)
    def model(x,w,b):
        return tf.matmul(x,w)+b
    predict: object=model(x,w,b)

"""
超参数
"""
train_epochs=70
learning_rate=0.01
"""
损失函数和优化器
"""

loss_fun=tf.reduce_mean(tf.square(y-predict))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fun)
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    loss_list=[]  #记录每轮的平均loss
    for epoch in range(train_epochs):
        loss_sum=0.0
        for xx,yy in zip(feature,label):
            xx=xx.reshape(1,12)
            yy=yy.reshape(1,1)
            _,loss=sess.run([optimizer,loss_fun],feed_dict={x:xx,y:yy})
            loss_sum=loss_sum+loss
        shuffle(feature,label)

        aver_loss = loss_sum / len(label)
        loss_list.append(aver_loss)

        print("当前轮次", epoch + 1, '平均损失值', aver_loss)
        # print("当前轮次",epoch + 1, '平均损失值',aver_loss,
        #       '\n当前w值\n', w.eval(),'当前b值', b.eval())


    n=np.random.randint(len(label))#从样本里随机选一个
    print("选择第%d号样品"%n)
    x_test=feature[n].reshape(1,12)
    predvalue=sess.run(predict,feed_dict={x:x_test})
    realvalue=label[n]
    print("predict:%f"%predvalue)
    print("real:",realvalue)
    plt.plot(loss_list)
    plt.show()






