import tensorflow as tf

import tensorflow.examples.tutorials.mnist.input_data as input




#读取数据集
mnist = input.read_data_sets("../DataSet/MNIST_data/", one_hot=True)

#定义一个占位符
x = tf.placeholder(tf.float32, [None, 784])

#定义权重值和偏置值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#实现模型 y为预测结果
y = tf.nn.softmax(tf.matmul(x,W) + b)

# y_代表正确结果
y_ = tf.placeholder("float", [None,10])

#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#梯度下降优化
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化所有的变量
init = tf.initialize_all_variables()
#生成回话

# config = tf.ConfigProto(
#         device_count = {'GPU': 1}
#     )

sess = tf.Session()
sess.run(init)

#训练1000次
for i in range(500):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))