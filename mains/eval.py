import math
import time
import sys
sys.path.append('./')
sys.path.append('../')
import numpy as np
import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from data_loader.data_generator import DataGenerator
from models import tdr2n2
from utils.config import process_config
from utils import deploy
from data_loader import util
from io import StringIO, BytesIO
import matplotlib.pyplot as plt
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

            x_test, y_test = data.get_eval_data()
            x_test = tf.expand_dims(x_test, -1)
            x_test.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
            y_test.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1]])
            y_test = tf.cast(y_test, tf.int32)
            y_test_hot = tf.one_hot(y_test, depth=config.network.num_classes, axis=-1)


        f_score, end_points = net.net(x_test)
        f_score_img = tf.expand_dims(tf.cast(tf.argmax(f_score, axis=-1), tf.float32)*50., -1)
        y_test_img = tf.expand_dims(tf.cast(tf.argmax(y_test_hot, axis=-1), tf.float32)*50., -1)


        ## add precision and recall
        f_score = tf.cast(tf.argmax(f_score, -1), tf.int32)
        #f_score = tf.image.resize_bilinear(f_score, (config.input.img_out_shape[0]))
        f_score = tf.one_hot(f_score, depth=config.network.num_classes, axis=-1)
        pred = tf.reduce_sum(f_score * y_test_hot, axis=(0, 1, 2))
        all_pred = tf.reduce_sum(f_score, axis=(0, 1, 2)) + 1e-5
        all_true = tf.reduce_sum(y_test_hot, axis=(0, 1, 2)) + 1e-5


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

        merged = tf.summary.merge_all()
        sum_writer = tf.summary.FileWriter(logdir=config.summary.test_dir)

        for checkpoint_path in evaluation.checkpoints_iterator(config.finetune.eval_checkpoint_dir):
            with tf.Session(config=configproto) as session:
                session.run(tf.global_variables_initializer())
                session.run(data.get_iterator(is_train=False).initializer)
                saver.restore(session, checkpoint_path)

                logging.info('Starting evaluation at ' + time.strftime(
                    '%Y-%m-%d-%H:%M:%S', time.gmtime()))
                k = 1
                tp = []
                tp_fp = []
                tp_fn = []
                imgs = []
                while True:
                    try:
                        pred_, all_pred_, all_true_, pred_img, true_img, g_step = session.run([pred, all_pred, all_true,
                                                                                            f_score_img, y_test_img,
                                                                                            global_step])
                        tp.append(np.expand_dims(pred_, 0))
                        tp_fp.append(np.expand_dims(all_true_, 0))
                        tp_fn.append(np.expand_dims(all_pred_, 0))
                        #img = util.merge_pics(pred_img, true_img)

                        print("deal with {} images".format(k*config.input.batch_size))
                        k += 1
                    except tf.errors.OutOfRangeError:
                        tp_ = np.sum(np.concatenate(tp, 0), 0)
                        tp_fn_ = np.sum(np.concatenate(tp_fn, 0), 0)
                        tp_fp_ = np.sum(np.concatenate(tp_fp, 0), 0)
                        precison = tp_ / tp_fp_
                        recall = tp_ / tp_fn_
                        dice = 2*tp_ /(tp_fp_+tp_fn_)

                        print(precison)
                        print(recall)
                        print(dice)
                        summary = tf.Summary()
                        for i in range(recall.shape[0]):
                            summary.value.add(tag='evaluation/{}th_class_precision'.format(i), simple_value=precison[i])
                            summary.value.add(tag='evaluation/{}th_class_recall'.format(i), simple_value=recall[i])
                            summary.value.add(tag='evaluation/{}th_class_dice'.format(i), simple_value=dice[i])
                        sum_writer.add_summary(summary, g_step)

                        break
                logging.info('Finished evaluation at ' + time.strftime(
                    '%Y-%m-%d-%H:%M:%S', time.gmtime()))




if __name__ == '__main__':
    tf.app.run()
