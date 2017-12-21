import tensorflow as tf

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# b_conv1 = bias_variable([10])

input = tf.Variable(tf.random_normal([1,64,64,5]))
filter = tf.Variable(tf.random_normal([3, 3, 5, 7]))
# (32-3)/1+1=30   -> (原始图像长-卷积核长)/步长+1=卷积后的长
op = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='VALID')

# bias=tf.nn.bias_add(op,b_conv1)
# h_conv1 = tf.nn.relu(bias)
# result=tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')VALID
print(op)

#
# a = tf.constant([
#     [[1.0, 2.0, 3.0, 4.0],
#      [5.0, 6.0, 7.0, 8.0],
#      [8.0, 7.0, 6.0, 5.0],
#      [4.0, 3.0, 2.0, 1.0]],
#     [[4.0, 3.0, 2.0, 1.0],
#      [8.0, 7.0, 6.0, 5.0],
#      [1.0, 2.0, 3.0, 4.0],
#      [5.0, 6.0, 7.0, 8.0]]
# ])
#
# a = tf.reshape(a, [1, 4, 4, 2])
#
# print(a)
#
# pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 1, 1, 1], padding='VALID')
# with tf.Session() as sess:
#     print("image:")
#     image = sess.run(a)
#     print(image)
#     print("reslut:",pooling)
#     result = sess.run(pooling)
#     print(result)