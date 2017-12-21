
import tensorflow as tf
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

from CNN.AlexNet import cifar10




def get_one_image(img_dir):
    image=Image.open(img_dir)
    # plt.imshow(image)
    # plt.show()

    image=image.resize([32,32])
    # plt.imshow(image)
    # plt.show()

    image_arr=np.array(image)



    return image_arr






def main(argv=None):
    log_dir='cifar10_train/'
    img_dir='cat.jpg'
    image_arr = get_one_image(img_dir)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 32, 32, 3])
        print(tf.shape(image))
        # plt.imshow(image)
        # plt.show()
        p = cifar10.inference(image)
        logits = tf.nn.softmax(p)

        print(tf.shape(p))
        x = tf.placeholder(tf.float32, shape=[32, 32, 3])

        variable_averages = tf.train.ExponentialMovingAverage(
                    cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')

            else:
                print('No checkpoint')
                prediction = sess.run(logits, feed_dict={x: image_arr})
                max_index = np.argmax(prediction)
                print(max_index)









    #
    # with tf.Graph.as_default(None):
    #     image_arr=get_one_image(img_dir)
    #
    #     image=tf.cast(image_arr,tf.float32)
    #
    #     image=tf.image.per_image_standardization(image)
    #
    #     image=tf.reshape(image,[1,32,32,3])
    #
    #     print(tf.shape(image))
    #     # cifar10.FLAGS.
    #
    #
    #     p=cifar10.inference(image)
    #
    #     logits=tf.nn.softmax(p)
    #
    #
    #     x=tf.placeholder(tf.float32,shape=[32,32,3])
    #     variable_averages = tf.train.ExponentialMovingAverage(
    #         cifar10.MOVING_AVERAGE_DECAY)
    #     variables_to_restore = variable_averages.variables_to_restore()
    #     saver=tf.train.Saver(variables_to_restore)
    #
    #     # with tf.Session as se:
    #     ckpt=tf.train.get_checkpoint_state(log_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #
    #         saver.restore(tf.Session,ckpt.model_checkpoint_path)
    #         print('load model success!!!!')
    #     else :
    #         print('load model faile!!!')
    #
    #         prediction=tf.Session.run(logits,feed_dict={x:image_arr})
    #
    #         max_index=tf.arg_max(prediction)
    #
    #         print(max_index)


if __name__ == '__main__':
    tf.app.run()

