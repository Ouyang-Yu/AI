# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 19:05
# @Author  : 欧阳煜
# @Email   : 2455356027@qq.com
# @File    : 4多分类逻辑回归_mnist单隐藏层.py

import tensorflow.compat.v1 as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_core.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("./data/",one_hot=True)
tf.logging.set_verbosity(old_v)

print("训练集数量:",mnist.train.num_examples)
print("验证集数量:",mnist.validation.num_examples)
print(mnist.test.num_examples)

feature=tf.placeholder(tf.float32, [None, 784])
label=tf.placeholder(tf.float32, [None, 10])



"""
图像加入summary,在tensorboard显示,一次10张
"""
image_shared_input=tf.reshape(feature,[-1,28,28,1])
tf.summary.image('input',image_shared_input,10)



H1_NN=256
H2_NN=64

def fcn_layer(inputs,   #输入的数据
              input_dim,#输入的神经元数量
              output_dim,#输出的神经元数量
              activation_fun=None):
    w=tf.Variable(tf.truncated_normal([input_dim,output_dim],stddev=0.1))
    b=tf.Variable(tf.zeros([output_dim]))
    ff=tf.matmul(inputs,w)+b
    if activation_fun is None:
        output=ff
    else:
        output=activation_fun(ff)
    return output



h1=fcn_layer(feature,784,H1_NN,tf.nn.relu)

h2=fcn_layer(h1,H1_NN,H2_NN,tf.nn.relu)

forward=fcn_layer(h2,H2_NN,10,None)


pred=tf.nn.softmax(forward)

"""
前向输出值在直方图中显示
"""
tf.summary.histogram("forward",forward)



"""
设置参数
"""
train_epoch=5
batch_size=50
total_batch=int(mnist.train.num_examples/batch_size)#一轮训练有多少批
display_step=1 #显示粒度
learning_rate=0.005

#交叉熵损失函数
# loss_fun=tf.reduce_mean(-tf.reduce_sum(label * tf.log(pred), reduction_indices=1))
loss_fun=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=label))

tf.summary.scalar("loss",loss_fun)#loss损失以标量显示


#避免log(0)造成nan数据不稳定,这里的forward是没有经过softmax()的
optiminizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_fun)

#定义准确率
correct_prediction=tf.equal(tf.argmax(pred,1), tf.arg_max(label, 1))
   #第二个参数1表示第二个维度,即列,游标在列之间移动,在每行中返回最大值的索引
#将布尔转化为浮点,并计算平均
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar("accuracy",accuracy)#accuracy以标量显示



save_step=5
import os
ckpt_dir="./ckpt/"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
saver=tf.train.Saver()
#开始训练之前,保存变量


with tf.Session() as sess:
    """
    在session中合并所有的smmary
    """
    merged_summary_op=tf.summary.merge_all()
    writer=tf.summary.FileWriter("log/",sess.graph)

    from time import time
    startTime=time()
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epoch):
        for batch in range(total_batch):
            xx,yy=mnist.train.next_batch(batch_size)
            sess.run(optiminizer,feed_dict={feature:xx,label:yy})
            #每训练一个批次就用writer写一次
            summary_str=sess.run(merged_summary_op,feed_dict={feature:xx,label:yy})
            writer.add_summary(summary_str,epoch)

        loss,acc=sess.run([loss_fun,accuracy],
                          feed_dict={feature:mnist.validation.images,
                                     label:mnist.validation.labels})

        if (epoch+1)%display_step==0:
            print("轮次:","%02d"%(epoch+1),"Loss=","{:.9f}".format(loss),
                  "Accuracy=","{:.4f}".format(acc))

        if (epoch + 1) % save_step == 0:#每轮训练结束后,选择保存
            saver.save(sess,os.path.join(ckpt_dir,
                                         "mnist_many_model_{:006}.ckpt".format(epoch+1)))
            print("mnist_many_model_{:006}.ckpt has saved".format(epoch+1))

    saver.save(sess,os.path.join(ckpt_dir,"mnist_many_model.ckpt"))
    print("Model saved")



    print("FINIDHED")
    duration=time()-startTime
    print("duration:",duration)


    accu_test=sess.run(accuracy,feed_dict={feature:mnist.test.images,
                                           label:mnist.test.labels})
    print("测试机的准确率",accu_test)
    pred_result = sess.run(tf.argmax(pred, 1), feed_dict={feature: mnist.test.images[:10]})
    print(pred_result[:10])  # test集合前十个识别结果

def plot_show(images,
              labels,
              prediction,
              index, #从index开始显示
              num=10):  #一次显示多少
    import matplotlib.pyplot as plt
    import numpy as np
    fig=plt.gcf()  #获取当前图表,getCurrntFigure
    fig.set_size_inches(10,12)
    if num>25:
        num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,i+1) #当前处理的子图
        ax.imshow(np.reshape(images[index],
                             (28,28)),
                  cmap="binary")
        title="label="+str(np.argmax(labels[index]))
        if len(prediction) >0:
            title+=",predict="+str(prediction[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])#不显示坐标轴
        index+=1
    plt.show()

plot_show(mnist.test.images,  mnist.test.labels,  pred_result,  0,  10)



