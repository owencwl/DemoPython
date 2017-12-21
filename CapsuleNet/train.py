import tensorflow as tf
from tqdm import tqdm

from CapsuleNet.config import cfg
from CapsuleNet.capsNet import CapsNet


if __name__ == "__main__":
     with tf.device('/gpu:0'):
        capsNet = CapsNet(is_training=True)
        tf.logging.info('Graph loaded')
        sv = tf.train.Supervisor(graph=capsNet.graph,
                             logdir='logdir',
                             save_model_secs=0)

        with sv.managed_session() as sess:
          num_batch = int(60000 / 128)
          for epoch in range(500):
            if sv.should_stop():
                break
            for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                sess.run(capsNet.train_op)

            global_step = sess.run(capsNet.global_step)
            sv.saver.save(sess, 'logdir' + '/model_epoch_%04d_step_%02d' % (epoch, global_step))

        tf.logging.info('Training done')
