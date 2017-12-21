import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input


#读取数据集
mnist = input.read_data_sets("../DataSet/MNIST_data/", one_hot=True)

#定义一个占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder("float", [None,10])

def weight_variable(shape):

#     shape: 一维的张量，也是输出的张量。
#     mean: 正态分布的均值。
#     stddev: 正态分布的标准差。
#     dtype: 输出的类型。
#     seed: 一个整数，当设置之后，每次生成的随机数都一样。
#     name: 操作的名字
#正态分布 stddev标准差
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



def conv2d(x, W):
  #   strides ：卷积时在图像每一维的步长，这是一个一维的向量，长度4 ；padding：string类型的量，只能是”SAME”,”VALID”其中之一。SAME以0填充边缘 VALID不填充
  # 结果返回的是Tensor，这个输出，就是我们常说的feature map
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max_pool
# 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
# 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
# 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]，strides参数确定了滑动窗口在各个维度上移动的步数
# strides[0] = 1，也即在 batch 维度上的移动为 1，也就是不跳过任何一个样本，否则当初也不该把它们作为输入（input）
# strides[3] = 1，也即在 channels 维度上的移动为 1，也就是不跳过任何一个颜色通道；
# 第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
# 返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
tf.device('/gpu:0')

# 相当于CNN中的卷积核，它要求是一个Tensor，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数] ，有32个卷积核
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape 举例 注意其中的-1
# # tensor 't' is [[[1, 1, 1],
# #                 [2, 2, 2]],
# #                [[3, 3, 3],
# #                 [4, 4, 4]],
# #                [[5, 5, 5],
# #                 [6, 6, 6]]]
# # tensor 't' has shape [3, 2, 3]
# # pass '[-1]' to flatten 't'
# reshape(t, [-1]) == > [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
#
# # -1 can also be used to infer the shape
#
# # -1 is inferred to be 9:
# reshape(t, [2, -1]) == > [[1, 1, 1, 2, 2, 2, 3, 3, 3],
#                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# # -1 is inferred to be 2:
# reshape(t, [-1, 9]) == > [[1, 1, 1, 2, 2, 2, 3, 3, 3],
#                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
# # -1 is inferred to be 3:
# reshape(t, [2, -1, 3]) == > [[[1, 1, 1],
#                               [2, 2, 2],
#                               [3, 3, 3]],
#                              [[4, 4, 4],
#                               [5, 5, 5],
#                               [6, 6, 6]]]
# 其中shape[-1,28,28,1]为一个列表形式，特殊的一点是列表中可以存在-1。
# -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1
# 变换为28*28的矩阵，最后的1为颜色通道
x_image = tf.reshape(x, [-1,28,28,1])
# 卷积
c_conv1=conv2d(x_image, W_conv1)
# 加上偏置
bias=tf.nn.bias_add(c_conv1,b_conv1)
# 将计算结果通过relu激活函数完成去线性化,这个函数的作用是计算激活函数relu，即max(features, 0)。即将矩阵中每行的非最大值置0。
h_conv1 = tf.nn.relu(bias)

# 使用最大池化
h_pool1 = max_pool_2x2(h_conv1)
# h_pool1 =tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1])
print(h_pool1)



W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)
h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, W_conv2) ,b_conv2))
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_flat, W_fc1) ,b_fc1))

# 为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
# TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。所以用dropout的时候可以不用考虑scale
keep_prob = tf.placeholder("float")
#keep_prob: 名字代表的意思, keep_prob 参数可以为 tensor，意味着，训练时候 feed 为0.5，测试时候 feed 为 1.0 就 OK。
#return：包装了dropout的x。训练的时候用，test的时候就不需要dropout了
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 添加一个softmax层，softmax模型可以用来给不同的对象分配概率
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 先加权求和，分别加上偏置，放入softmax中
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)



#分类问题损失函数：交叉熵，交叉熵刻画了两个概率分布之间的距离
# 回归问题损失函数：均方误差（MSE，mean squared error）；回归问题是对具体数值的预测，即预测的是任意实数
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
#使用优化算法使得代价函数最小化
# Optimizer
# GradientDescentOptimizer
# AdagradOptimizer
# AdagradDAOptimizer
# MomentumOptimizer
# AdamOptimizer
# FtrlOptimizer
# RMSPropOptimizer
train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

#找出预测正确的标签
# argmax函数是对矩阵按行或列计算最大值
# input：输入Tensor
# axis：0表示按列，1表示按行
# name：名称
# dimension：和axis功能一样，默认axis取值优先。新加的字段
# 返回：Tensor  一般是行或列的最大值下标向量
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
#得出通过正确个数除以总数得出准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# 使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，只是我们会用更加复杂的ADAM优化器来做梯度最速下降，
# 在feed_dict中加入额外的参数keep_prob来控制dropout比例。然后每100次迭代输出一次日志。
for i in range(10000):
  batch = mnist.train.next_batch(50)
  # 开始训练
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
  # 每50次 验证一次
  if i%50 == 0:
      #   feed_dict用来给占位符 赋值 ，batch[0]为图片数据，batch[1] 为类别数据，keep_prob：dropout率
      train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %.5f"%(i, train_accuracy))



print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
