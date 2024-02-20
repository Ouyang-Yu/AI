import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

x_dataSet=np.linspace(-1, 1, 100)   #100个 -1...1
print(x_dataSet.shape) #(100,) 一维数组,100个元素 shape是元组,加星号拆包,提出参数
print(*x_dataSet.shape) #100

y_dataSet= 2 * x_dataSet + 1 + np.random.randn(*x_dataSet.shape) * 0.4


train_epoch=20  #轮次
learning_rate=0.02

feature=tf.placeholder(tf.float32)
label=tf.placeholder(tf.float32)



w=tf.Variable(1.0)
b=tf.Variable(0.0)


loss_fun=tf.reduce_mean(tf.square(label - (w * feature + b)))
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fun)  #优化器优化均方差损失函数

with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(train_epoch):
        for xx,yy in zip(x_dataSet, y_dataSet):
            _, loss=sess.run([optimizer,loss_fun], feed_dict={feature:xx, label:yy})
            b_trained=b.eval()
            w_trained=w.eval()
        #训练过程可视化
        plt.plot(x_dataSet, b_trained + w_trained * x_dataSet)
        print("训练轮次:",epoch+1,"  目前的参数值w:",w_trained,"   b:",b_trained)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("训练过程中的图像")
    plt.show()

    finalb = sess.run (b)
    finalw = sess.run(w)
    print("final w",finalw)
    print("final b",finalb)
    print("3.0  的  predict",3*w.eval()+b.eval())  #测一个点试试

"""
展示最终结果可视化
"""
plt.scatter(x_dataSet, y_dataSet, label="original")
plt.plot(x_dataSet, x_dataSet * finalw + finalb, label="fitted")
plt.legend(loc=2) #图例
plt.title("散点图与最后的训练结果")
plt.show()



