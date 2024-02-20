import tensorflow.compat.v1 as  tf
tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
#读取数据文件
df = pd.read_csv('boston.csv', header=0)
#显示数据摘要描述信息
#print(df.describe())
df = df.values
df = np.array(df)#转化成数组的形式
#对特征数据【0,11】做归一化处理
for i in range(12):
    df[:,i] = (df[:,i]-df[:,i].min())/(df[:,i].max()-df[:,i].min())
x_data = df[:,:12]#前12列特征数据
y_data = df[:,12]#最后一列标签数据


x = tf.placeholder(tf.float32,[None,12],name='X')#12个特征数据（12列）
y = tf.placeholder(tf.float32,[None,1],name='Y')#1个标签数据（1列）


#定义命名空间
with tf.name_scope('Model'):
    #初始化值为shape=（12,1）的随机数
    w = tf.Variable(tf.random_normal([12,1],stddev=0.01),name='W')
    #b初始化值为1.0
    b = tf.Variable(1.0,name='b')
    #w和x是矩阵相称，用matmul

    #预测计算操作，前向计算节点



#设置超参数
train_epochs = 20#迭代轮数
learning_rate = 0.01#学习率



#定义均方损失函数
with tf.name_scope('LossFunction'):
    loss_function = tf.reduce_mean(tf.pow(y-tf.matmul(x,w)+b,2))#均方误差


#创建优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)
#声明会话
sess = tf.Session()
#设置日志存储目录
logdir = 'D:/dream/TensorFlow/log'
#创建一个操作，用于记录损失值loss
sum_loss_op = tf.summary.scalar('loss',loss_function)
#把所有需要记录摘要日志文件的合并，方便一次性写入
merged = tf.summary.merge_all()
#定义初始化变量的操作



init = tf.global_variables_initializer()
#启动会话
sess.run(init)#初始化需要run一遍
#创建摘要的文件写入器（FileWriter）#将计算图写入
writer = tf.summary.FileWriter(logdir,sess.graph)
#迭代训练
loss_list = []#用于保存loss值的列表
for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs,ys in zip(x_data,y_data):
        #数据变形
        xs = xs.reshape(1,12)
        ys = ys.reshape(1, 1)
        _,summary_str,loss = sess.run([optimizer,sum_loss_op,loss_function],feed_dict={x:xs,y:ys})
        writer.add_summary(summary_str,epoch)
        #计算本轮loss值的和
        loss_sum = loss_sum + loss
    #打乱数据顺序
    xvalues,yvalues = shuffle(x_data,y_data)
    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)
    loss_list.append(loss_average)#每轮添加一次
    print(epoch+1,loss_average,b0temp,w0temp)


n = np.random.randint(506)
print(n)
x_test = x_data[n]
print(x_test)
x_test = x_test.reshape(1,12)
predict = sess.run(tf.matmul(x,w)+b,feed_dict={x:x_test})
print('预测值：%f' % predict)
target = y_data[n]
print('标签值：%f' % target)
plt.plot(loss_list)
plt.show()






