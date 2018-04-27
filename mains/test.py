import math
import time
import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.platform import tf_logging as logging
from data_loader.data_generator_lstm import DataGenerator
from models import tdr2n2
from utils.config import process_config
from utils import deploy
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'config_path', '../configs/example.json',
    'configuration file path.')


FLAGS = tf.app.flags.FLAGS


def main(_):

    config = process_config(FLAGS.config_path)
    print(config)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        deploy_config = deploy.DeploymentConfig(
                        num_clones=1)

        global_step = tf.Variable(0,trainable=False, name='global_step')

        # select model and build net
        net = tdr2n2.Unet(config)

        # create batch dataset
        with tf.device(deploy_config.inputs_device()):
            data = DataGenerator(config.input)

            x_test, y_test = data.get_test_data()
            x_test = tf.expand_dims(x_test, -1)
            x_test.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
            y_test.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1]])
            y_test = tf.cast(y_test, tf.int32)
            y_test_hot = tf.one_hot(y_test, depth=config.network.num_classes, axis=-1)


        f_score, end_points = net.net(x_test)
        f_score_img = tf.expand_dims(tf.cast(tf.argmax(f_score, axis=-1), tf.float32), -1)

        # Variables to restore: moving avg. or normal weights.
        if config.train.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                config.train.moving_average_decay, global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                slim.get_model_variables())
            variables_to_restore[global_step.op.name] = global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        saver = None
        if variables_to_restore is not None:
            saver = tf_saver.Saver(variables_to_restore)

        # =================================================================== #
        # Evaluation loop.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.deploy.gpu_memory_fraction)
        configproto = tf.ConfigProto(gpu_options=gpu_options,
                                     log_device_placement=False,
                                     allow_soft_placement=True, )


        with tf.Session(config=configproto) as session:
                session.run(tf.global_variables_initializer())
                #session.run(data.get_iterator(is_train=False).initializer)
                saver.restore(session, config.finetune.checkpoint_path)

                logging.info('Starting evaluation at ' + time.strftime(
                    '%Y-%m-%d-%H:%M:%S', time.gmtime()))
                i = 0
                while True:
                    try:
                        pred_img, _ = session.run([f_score_img, global_step])
                        for img in pred_img:
                            img[img>5] = 0
                            print(np.unique(img))
                            img *= 50
                            cv2.imwrite('../data/aapm_new/test/result/{}.png'.format(i), img)
                            i += 1
                            print(i)
                    except tf.errors.OutOfRangeError:
                        break
                logging.info('Finished evaluation at ' + time.strftime(
                    '%Y-%m-%d-%H:%M:%S', time.gmtime()))




if __name__ == '__main__':
    tf.app.run()
