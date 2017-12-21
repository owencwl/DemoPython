import os
import scipy
import numpy as np
import tensorflow as tf

from CapsuleNet.config import cfg


def load_mnist(path, is_training):
    # fd = open(r'..\DataSet\MNIST_data\train-images-idx3-ubyte\train-images-idx3-ubyte')
    fd = open(os.path.join(r'../DataSet/MNIST_data', 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(r'../DataSet/MNIST_data', 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(r'../DataSet/MNIST_data', 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(r'../DataSet/MNIST_data', 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    # normalization and convert to a tensor [60000, 28, 28, 1]
    trX = tf.convert_to_tensor(trX / 255., tf.float32)

    # => [num_samples, 10]
    trY = tf.one_hot(trY, depth=10, axis=1, dtype=tf.float32)
    teY = tf.one_hot(teY, depth=10, axis=1, dtype=tf.float32)

    if is_training:
        return trX, trY
    else:
        return teX / 255., teY


def get_batch_data():
    trX, trY = load_mnist(r'..\DataSet\MNIST_data', True)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=8,
                                  batch_size=128,
                                  capacity=128 * 64,
                                  min_after_dequeue=128 * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


if __name__ == '__main__':
    X, Y = load_mnist(cfg.dataset, cfg.is_training)
    print(X.get_shape())
    print(X.dtype)
