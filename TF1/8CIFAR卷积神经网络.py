# -*- coding: utf-8 -*-
# @Time    : 2020/1/15 11:44
# @Author  : 欧阳煜
# @Email   : 2455356027@qq.com
# @File    : 8CIFAR卷积神经网络.py

import os
import numpy as np
import pickle

def load_cifar_batch(filename):
    with open(filename, "rb") as f:
        data_dict=pickle.load(f,encoding="bytes")
        images=data_dict[b'data']
        labels=data_dict[b'labels']

        images=images.reshape(10000,3,32,32)
        images=images.transpose(0,2,3,1)#把数据通道c移动到最后一个维度
        labels=np.array(labels)
        return images,labels
def load_cifar_data(data_dir):
    images_train=[]
    labels_train=[]
    for i in range(5):
        f=os.path.join(data_dir,'data_batch_%d'%(i+1))
        print("loading",f)
        image_batch,label_batch=load_cifar_batch(f)
        images_train.append(image_batch)
        labels_train.append(label_batch)
        Xtrain=np.concatenate(images_train)
        Ytrain=np.concatenate(labels_train)
        del image_batch,label_batch
    Xtset,Ytest=load_cifar_batch(os.path.join(data_dir,'test_batch'))
    print('finished load CIFAR')
    return Xtrain,Ytrain,Xtset,Ytest#返回测试集的图像和标签和训练集的图像和标签
data_dir='data/cifar-10-batches-py//'
Xtrain,Ytrain,Xtest,Ytest=load_cifar_data(data_dir)

print("train images shape",Xtrain.shape)
print("train labels shape",Ytrain.shape)
print("test images shape",Xtest.shape)
print("test labels shape",Ytest.shape)

import matplotlib.pyplot as plt
# plt.imshow(Xtrain[3333])
# plt.show()
# print(Ytrain[456])

label_dict={0:'airplane',
            1:'automobile',
            2:'brid',
            3:'cat',
            4:'deer',
            5:'dog',
            6:'frog',
            7:'horse',
            8:'ship',
            9:'tunck'
            }
def plot_images(images,labels,prediction,index,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12,6)
    if num>10:
        num=10
    for i in range(num):
        ax=plt.subplot(2,5,i+1)
        ax.imshow(images[index],cmap='binary')
        title=str(i)+','+label_dict[labels[index]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[index]]
        ax.set_title(title, fontsize=10)
        index+=1
    plt.show()
    
plot_images(Xtest,Ytest,[],50,10)

"""
数据预处理
"""

Xtarin_normalize=Xtrain.astype('float32')/255
Xtest_normolize=Xtest.astype('float32')/255  #将一个像素点[255,555,255]标准化

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(sparse=False)
"""
标签热编码
"""
yy=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
encoder.fit(yy)
Ytarin_reshape=Ytrain.reshape(-1,1)
YTrain_oneHot=encoder.transform(Ytarin_reshape)
Ytest_reshape=Ytest.reshape(-1,1)
Ytest_onehot=encoder.transform(Ytest_reshape)

# print(Ytest[1],Ytest_onehot[1])



import tensorflow.compat.v1 as tf
tf.reset_default_graph()

"""
定义共享函数
"""
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1),name='W')
def bias(shape):
    return tf.Variable(tf.constant(0.1,shape=shape),name='b')
def conv2d(x,W):
    """
    定义卷积,步长为一,零填充
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    """
    池化,步长为2,即原尺寸长宽各除以2
    """
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


"""
定义网络结构
"""
with tf.name_scope("input_layer"):
    x=tf.placeholder('float',shape=[None,32,32,3],name="X")

with tf.name_scope('conv_1'):
    W1=weight([3,3,3,32])#卷积核长宽,输入通道数,输出通道数
    b1=bias([32])#与输出通道数一致
    conv_1=tf.nn.relu(conv2d(x,W1)+b1)

with tf.name_scope("pool_1"):
    pool_1=max_pool(conv_1)

with tf.name_scope('conv_2'):
    W2=weight([3,3,32,64])#卷积核长宽,输入通道数,输出通道数
    b2=bias([64])#与输出通道数一致
    conv_2=tf.nn.relu(conv2d(pool_1,W2)+b2)
with tf.name_scope("pool_2"):
    pool_2=max_pool(conv_2)

with tf.name_scope("fc"):
    w3=weight([4096,128])
    b3=bias([128])
    flat=tf.reshape(pool_2,[-1,4096])
    h=tf.nn.relu(tf.matmul(flat,w3)+b3)
    h_dropout=tf.nn.dropout(h,keep_prob=0.8) #避免过拟合
with tf.name_scope("output_layer"):
    w4=weight([128,10])
    b4=bias([10])
    pred=tf.nn.softmax(tf.matmul(h_dropout,w4)+b4)


"""
构建模型
"""
with tf.name_scope("optimizer"):
    y=tf.placeholder('float',shape=[None,10],name="label")
    loss_fun=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
    optimizer=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_fun)

with tf.name_scope("evaluation"):
    correct_pridiction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pridiction,'float'))


"""
训练模型
"""
import os
from time import time

train_epochs=25
batch_size=50

total_batch=int(len(Xtrain)/batch_size)
epoch_list=[];accuracy_list=[];loss_list=[]

epoch=tf.Variable(0,name='epoch',trainable=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    """
    断点续训
    """
    ckpt_dir="./CIFAR_log/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if checkpoint is not None:
        saver=tf.train.Saver(max_to_keep=1)
        saver.restore(sess,checkpoint)
    else:
        print('trian form scratch')
    start=sess.run(epoch)
    print('training from {} epoch'.format(start+1))


    """
    迭代训练
    """
    starttime=time()
    def get_train_batch(number,batch_size):
        return Xtarin_normalize[number*batch_size:(number+1)*batch_size], \
               YTrain_oneHot[number * batch_size:(number + 1) * batch_size]

    for ep in range(start,train_epochs):
        for i in range(total_batch):
            batch_x,batch_y=get_train_batch(i,batch_size)
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y})
            if i%100==0:
                print("step {}".format(i),"finished")
        loss,acc=sess.run([loss_fun,accuracy],feed_dict={x:batch_x,y:batch_y})
        epoch_list.append(ep + 1)
        loss_list.append(loss)
        accuracy_list.append(acc)

        print("train epoch:",'%02d'%(sess.run(epoch)+1),
              "Loss:","{:.6f}".format(loss),"Accuracy:",acc)

        #保存检查点
        saver.save(sess,ckpt_dir+"CIFAR10_cnn_model.ckpt",global_step=ep+1)
        sess.run(epoch.assign(ep+1))
        print("已保存当前轮次训练结果")
    duration=time()-starttime
    print('duration',duration)

    #可视化损失值
    fig=plt.gcf()
    fig.set_size_inches(4,2)
    plt.plot(epoch_list, loss_list, label='loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss'],loc='upper right')


    #可视化准确率
    plt.plot(epoch_list, accuracy_list,label='accuracy')
    fig=plt.gcf()
    fig.set_size_inches(4,2)
    plt.ylim(0.1,1)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()





        









