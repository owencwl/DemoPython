import tensorflow as tf
import sys

flags=tf.app.flags

cfg = tf.app.flags.FLAGS

############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 500, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')


############################
#   environment setting    #
############################
flags.DEFINE_string('dataset', r'..\DataSet\MNIST_data', 'the path for dataset')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
flags.DEFINE_string('logdir', 'logdir', 'logs directory')




def main(unused_argv):



    tf.logging.set_verbosity(tf.logging.INFO)

    test=cfg.dataset
    print(test)

    # print(cfg.get_flag_value(name='dataset',default=None))

if __name__ =='__main__':

    tf.app.run()

