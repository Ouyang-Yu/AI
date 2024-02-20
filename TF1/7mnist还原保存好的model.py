# -*- coding: utf-8 -*-
# @Time    : 2020/1/14 21:12
# @Author  : 欧阳煜
# @Email   : 2455356027@qq.com
# @File    : 7mnist还原保存好的model.py
import tensorflow.compat.v1 as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow_core.examples.tutorials.mnist.input_data as input_data

mnist=input_data.read_data_sets("./data/",one_hot=True)
tf.logging.set_verbosity(old_v)


feature=tf.placeholder(tf.float32, [None, 784])
label=tf.placeholder(tf.float32, [None, 10])

H1_NN=256
H2_NN=64

def fcn_layer(inputs,   #输入的数据
              input_dim,#输入的神经元数量
              output_dim,#输出的神经元数量
              activation_fun=None):
    """"""
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
定义相同结构的模型
"""



"""
设置参数
"""
# train_epoch=10
# batch_size=10
# total_batch=int(mnist.train.num_examples/batch_size)#一轮训练有多少批
# display_step=1 #显示粒度
# learning_rate=0.01

#交叉熵损失函数
# loss_fun=tf.reduce_mean(-tf.reduce_sum(label * tf.log(pred), reduction_indices=1))
# loss_fun=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=forward, labels=label))
#避免log(0)造成nan数据不稳定,这里的forward是没有经过softmax()的
# optiminizer=tf.train.AdamOptimizer(learning_rate).minimize(loss_fun)

# 定义准确率
correct_prediction=tf.equal(tf.argmax(pred,1), tf.arg_max(label, 1))
#    第二个参数1表示第二个维度,即列,游标在列之间移动,在每行中返回最大值的索引
# 将布尔转化为浮点,并计算平均
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    saver=tf.train.Saver()

    ckpt_dir= "./ckpt/"
    sess.run(tf.global_variables_initializer())

    ckpt_state=tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt_state and ckpt_state.model_checkpoint_path:
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        print("restore model from " + ckpt_state.model_checkpoint_path)



    print("Accuracy:",accuracy.eval(session=sess,
                                    feed_dict={feature:mnist.test.images,
                                               label:mnist.test.labels}))
    pred_result = sess.run(tf.argmax(pred, 1), feed_dict={feature: mnist.test.images[:]})
    print(str(pred_result))  # test集合前十个识别结果
    print(sess.run(tf.argmax(mnist.test.labels[0])))


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
    #pred_result=[7 2 1 0 4 1 4 9 5 9]
    def show_pred(k):


        import matplotlib.pyplot as plt
        import numpy as np
        fig = plt.gcf()  # 获取当前图表,getCurrntFigure
        fig.set_size_inches(10, 12)

        mypred = sess.run(tf.argmax(pred, 1), feed_dict={feature: mnist.test.images[k - 1:k]})
        actual = sess.run(tf.argmax(mnist.test.labels[k - 1]))
        ax = plt.subplot(1,1,1)  # 当前处理的子图
        ax.imshow(np.reshape(mnist.test.images[k-1],
                             (28, 28)),
                  cmap="binary")
        title = "label=" + str(actual)

        title += ",predict=" + str(mypred)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])  # 不显示坐标轴
        plt.show()

    while True:
        input_number = input("Enter the number you want to see in mnist(from 1)\n")
        if input_number.isdigit():
            aaa = int(input_number)
        else:
            print("not number")
            continue
        show_pred(aaa)