import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
#
# import tensorflow_core.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./data/",one_hot=True)
tf.logging.set_verbosity(old_v)

print("训练集数量:",mnist.train.num_examples)
print("验证集数量:",mnist.validation.num_examples)
print(mnist.test.num_examples)
feature=tf.placeholder(tf.float32, [None, 784])
label=tf.placeholder(tf.float32, [None, 10])
w=tf.Variable(tf.random_normal([784,10]))
b=tf.Variable(tf.zeros([10]))

forward= tf.matmul(feature, w) + b  #单个神经元计算结果:一维数组,十个值
probability=tf.nn.softmax(forward)#过softmax得到分类概率

"""
设置参数
"""
train_epoch=20
batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)#一轮训练有多少批
display_step=1 #显示粒度
learning_rate=0.01

#交叉熵损失函数
loss_fun=tf.reduce_mean(-tf.reduce_sum(label * tf.log(probability), reduction_indices=1))
optiminizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_fun)

#定义准确率
correct_prediction=tf.equal(tf.argmax(probability, 1), tf.arg_max(label, 1))
   #第二个参数1表示第二个维度,即列,在每行中返回最大值的那一列

#将布尔转化为浮点,并计算平均
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    init=tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(train_epoch):
        for batch in range(total_batch):
            xx,yy=mnist.train.next_batch(batch_size)
            sess.run(optiminizer,feed_dict={feature:xx,label:yy})

        loss,acc=sess.run([loss_fun,accuracy],
                          feed_dict={feature:mnist.validation.images,
                                     label:mnist.validation.labels})

        if (epoch+1)%display_step==0:
            print("轮次:","%02d"%(epoch+1),"Loss=","{:.9f}".format(loss),
                  "Accuracy=","{:.4f}".format(acc))
    print("FINIDHED")


    accu_test=sess.run(accuracy,feed_dict={feature:mnist.test.images,
                                           label:mnist.test.labels})
    print("测试机的准确率",accu_test)


    pred_result=sess.run(tf.argmax(probability, 1), feed_dict={feature: mnist.test.images[:10]})
    print(pred_result[:10])  #test集合前十个识别结果





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







