
import gzip
import numpy as np
import os
import tensorflow as tf
import pickle
import time

LABEL_SIZE = 1
IMAGE_SIZE = 32
NUM_CHANNELS = 3
PIXEL_DEPTH = 255
NUM_CLASSES = 10

TRAIN_NUM = 10000
TRAIN_NUMS = 50000
TEST_NUM = 10000


# 接着我们定义提取数据的函数



def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f,encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32,32).transpose(0,2,3,1).astype("float32")
    Y = np.array(Y)
    return X, Y

def extract_data(filenames):
    # 验证文件是否存在
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 读取数据
    images = []
    labels = []
            # 读取数据
        # labels = None
        # images = None

    for f in filenames:
        X,Y=load_CIFAR_batch(f)
        images.append(X)
        labels.append(Y)
    image_r=np.concatenate(images)
    label_r=np.concatenate(labels)



    #     bytestream = open(f, 'rb')
    #     # 读取数据，首先将数据集中的数据读取进来作为buf
    #     buf = bytestream.read(TRAIN_NUM * (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS + LABEL_SIZE))
    #     # 把数据流转化为np的数组,为什么要转化为np数组呢，因为array数组只支持一维操作，为了满足我们的操作需求，我们利用np.frombuffer()将buf转化为numpy数组现在data的shape为（30730000，），3073是3*1024+1得到的，3个channel（r，g，b），每个channel有1024=32*32个信息，再加上 1 个label
    #
    #     data = np.frombuffer(buf, dtype=np.uint8)
    #
    #     # 改变数据格式,将shape从原来的（30730000，）——>为（10000，3073）
    #     data = data.reshape(TRAIN_NUM, LABEL_SIZE + IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)
    #
    #     # 分割数组,分割数组，np.hsplit是在水平方向上，将数组分解为label_size的一部分和剩余部分两个数组，在这里label_size=1，也就是把标签label给作为一个数组单独切分出来如果你对np.split还不太了解，可以自行查阅一下，此时label_images的shape应该是这样的[array([.......]) , array([.......................])]
    #     labels_images = np.hsplit(data, [LABEL_SIZE])
    #
    #     label = labels_images[0].reshape(TRAIN_NUM)
    #     # 此时labels_images[0]就是我们上面切分数组得到的第一个数组，在这里就是label数组，这时的shape为array([[3] , [6] , [4] , ....... ,[7]])，我们把它reshape（）一下变为了array([3 , 6 , ........ ,7])
    #
    #     image = labels_images[1].reshape(TRAIN_NUM, IMAGE_SIZE, IMAGE_SIZE,NUM_CHANNELS)
    #     # 此时labels_image[1]就是我们上面切分数组的剩余部分，也就是图片部分我们把它reshape（）为（10000，32，32，3）
    #
    #     if labels == None:
    #         labels = label
    #         images = image
    #     else:
    #         # 合并数组，不能用加法
    #         labels = np.concatenate((labels, label))
    #         images = np.concatenate((images, image))
    #
    # images = (images - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    #
    # return labels, images
    return label_r,image_r

# 定义提取训练数据函数


def extract_train_data(files_dir):
    # 获得训练数据
    # filenames = [os.path.join(files_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    filenames = [os.path.join(files_dir, 'data_batch_%d' % i) for i in range(1, 6)]
    return extract_data(filenames)

# 定义提取测试数据函数


def extract_test_data(files_dir):
    # 获得测试数据
    # filenames = [os.path.join(files_dir, 'test_batch.bin'), ]
    filenames = [os.path.join(files_dir, 'test_batch'), ]
    return extract_data(filenames)

# 把稠密数据label[1, 5...]
# 变为[[0, 1, 0, 0...], [...]...]


def dense_to_one_hot(labels_dense, num_classes):
    # 数据数量,np.shape[0]返回行数，对于一维数据返回的是元素个数,如果读取了5个文件的所有训练数据，那么现在的num_labels的值应该是50000
    num_labels = labels_dense.shape[0]

    # 生成[0,1,2...]*10,[0,10,20...],之所以这样子是每隔10个数定义一次值，比如第0个，第11个，第22个......的值都赋值为1
    index_offset = np.arange(num_labels) * num_classes

    # 初始化np的二维数组,一个全0，shape为(50000,10)的数组
    labels_one_hot = np.zeros((num_labels, num_classes))

    # 相对应位置赋值变为[[0,1,0,0...],[...]...],np.flat将labels_one_hot砸平为1维
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

# 定义cifar10数据集类

class Cifar10DataSet(object):
    """docstring for Cifar10DataSet"""

    def __init__(self, data_dir):
        super(Cifar10DataSet, self).__init__()
        self.train_labels, self.train_images = extract_train_data(
            data_dir)
        self.test_labels, self.test_images = extract_test_data(data_dir)

        print(self.train_labels.size) #50000
        print(self.test_labels.size)  #10000

        self.train_labels = dense_to_one_hot(self.train_labels, NUM_CLASSES)
        self.test_labels = dense_to_one_hot(self.test_labels, NUM_CLASSES)

        # epoch完成次数
        self.epochs_completed = 0
        # 当前批次在epoch中进行的进度
        self.index_in_epoch = 0

    def next_train_batch(self, batch_size):
        # 起始位置
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        # print "self.index_in_epoch: ",self.index_in_epoch
        # 完成了一次epoch
        if self.index_in_epoch > TRAIN_NUMS:
            # epoch完成次数加1,50000张全部训练完一次，那么没有数据用了怎么办，采取的办法就是将原来的数据集打乱顺序再用
            self.epochs_completed += 1
            # print "self.epochs_completed: ",self.epochs_completed
            # 打乱数据顺序，随机性
            perm = np.arange(TRAIN_NUMS)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]
            start = 0
            self.index_in_epoch = batch_size
            # 条件不成立会报错
            assert batch_size <= TRAIN_NUMS

        end = self.index_in_epoch
        # print "start,end: ",start,end

        return self.train_images[start:end], self.train_labels[start:end]

    def test_data(self):
        return self.test_images, self.test_labels



def main():
    cc = Cifar10DataSet('../DataSet/cifar-10-batches-py')
    # cc = input.read_data_sets("../DataSet/MNIST_data/", one_hot=True)
    # X,Y=cc.next_train_batch(1)

    tf.device("")
    X = tf.placeholder(tf.float32, [None, 32 ,32,3])
    y_ = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)


    reshaped_image = tf.cast(X, tf.float32)
    X = tf.reshape(reshaped_image, [-1, 32, 32, 3])


    with tf.variable_scope('layer1_conv1') as  scope:
        conv1_weight=tf.get_variable('weight',[3, 3, 3, 48],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases=tf.get_variable('biases',48,initializer=tf.constant_initializer(0.1))
        conv1=tf.nn.conv2d(X,conv1_weight,strides=[1, 1, 1, 1], padding='SAME')
        print(conv1)
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases),name=scope.name)

        pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
        # norm1
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        # tf.summary.histogram(scope.name,norm1)
        print(norm1)

    with tf.variable_scope('layer2_conv2') as  scope:
        conv2_weight = tf.get_variable('weight', [3, 3, 48, 96], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases', 96, initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(norm1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        print(conv2)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases), name=scope.name)
        # norm2
        norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.summary.histogram(scope.name, pool2)


    with tf.variable_scope('layer3_conv3') as  scope:
        conv3_weight = tf.get_variable('weight', [3, 3, 96, 128],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable('biases', 128, initializer=tf.constant_initializer(0.1))

        conv3= tf.nn.conv2d(pool2, conv3_weight, strides=[1, 1, 1, 1], padding='SAME')
        print(conv3)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases), name=scope.name)
        # norm2
        norm3 = tf.nn.lrn(relu3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.summary.histogram(scope.name, pool3)


    with tf.variable_scope('layer4_conv4') as  scope:
        conv4_weight = tf.get_variable('weight', [3, 3, 128, 256],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable('biases', 256, initializer=tf.constant_initializer(0.1))
        conv4 = tf.nn.conv2d(pool3, conv4_weight, strides=[1, 1, 1, 1], padding='SAME')
        print(conv4)
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases), name=scope.name)
        # norm2
        norm4 = tf.nn.lrn(relu4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.summary.histogram(scope.name, pool4)

    with tf.variable_scope('layer5_conv5') as  scope:
        conv5_weight = tf.get_variable('weight', [3, 3, 256, 512],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv5_biases = tf.get_variable('biases', 512, initializer=tf.constant_initializer(0.1))
        conv5 = tf.nn.conv2d(pool4, conv5_weight, strides=[1, 1, 1, 1], padding='SAME')
        print(conv5)
        relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases), name=scope.name)
        # norm2
        norm5 = tf.nn.lrn(relu5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2')
        pool5 = tf.nn.max_pool(norm5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.summary.histogram(scope.name, pool5)
        print(pool5)

        reshaped=tf.reshape(pool5,[-1,1*1*512])

    with tf.variable_scope('full_contact1') as  scope:
        fc1_weight=tf.get_variable('weight',[512,384],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases=tf.get_variable('biases',[384],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_weight)+fc1_biases)
        # tf.summary.histogram(scope.name, fc1)
        print(fc1)


    with tf.variable_scope('full_contact2') as  scope:
        fc2_weight = tf.get_variable('weight', [384,192], initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1))
        fc2=tf.nn.relu(tf.matmul(fc1,fc2_weight)+fc2_biases)
        print(fc2)
        fc1_drop=tf.nn.dropout(fc2,keep_prob)
        # tf.summary.histogram(scope.name, fc1_drop)


    with tf.variable_scope('full_contact3') as  scope:
        W = tf.get_variable('weight', [192, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.1))
        y_conv = tf.maximum(tf.nn.softmax(tf.matmul(fc1_drop, W) + b), 1e-30)
        # tf.summary.histogram(scope.name, y_conv)



    cross_entropy = -tf.reduce_mean(y_ * tf.log(y_conv))

    # tf.summary.scalar("loss",cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess = tf.InteractiveSession()

    # merged=tf.summary.merge_all()
    # writer=tf.summary.FileWriter(r"E:\tensorboard",sess.graph)

    sess.run(tf.global_variables_initializer())

    start_time = time.time()
    for i in range(10000):
        batch = cc.next_train_batch(50)
        # 开始训练
        train_step.run(feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
        # 每50次 验证一次
        if i % 50 == 0:
            #   feed_dict用来给占位符 赋值 ，batch[0]为图片数据，batch[1] 为类别数据，keep_prob：dropout率
            train_accuracy = accuracy.eval(feed_dict={X: batch[0], y_: batch[1], keep_prob: 1.0})
            # tf.summary.scalar("accuracy",train_accuracy)

            # result=sess.run(merged,feed_dict={X: batch[0], y_: batch[1], keep_prob: 0.5})
            # writer.add_summary(result,i)

            print("step %d, training accuracy %.5f" % (i, train_accuracy))
            # 计算间隔时间
            end_time = time.time()
            print('time: ', (end_time - start_time))
            start_time = end_time

    print("test accuracy %g" % accuracy.eval(feed_dict={
        X: cc.test_images, y_: cc.test_labels, keep_prob: 1.0}))



if __name__ == '__main__':
    with tf.device('/gpu:0'):
       main()

