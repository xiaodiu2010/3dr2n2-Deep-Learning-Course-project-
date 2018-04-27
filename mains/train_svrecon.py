import tensorflow as tf
import sys
sys.path.append('./')
sys.path.append('../')
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from models.model_factory import get_model_fn
from data_loader.generator_factory import get_generator_fn
from models import tdr2n2
from utils.config import process_config
from utils import deploy
from utils import tf_utils
from data_loader import util
slim = tf.contrib.slim

tf.app.flags.DEFINE_string(
    'config_path', '../configs/ae.json',
    'configuration file path.')


FLAGS = tf.app.flags.FLAGS



def main(_):
    # capture the config path from the run arguments
    # then process the json configration file
    config = process_config(FLAGS.config_path)
    print(config)

    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Graph().as_default():
        ######################
        # Config model_deploy#
        ######################
        deploy_config = deploy.DeploymentConfig(
            num_clones=config.deploy.num_clone)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = tf.Variable(0, trainable=False, name='global_step')

        # select model and build net
        net = get_model_fn(config)(config)

        # create batch dataset
        with tf.device(deploy_config.inputs_device()):
            generator = get_generator_fn(config)
            data = generator(config.input)
            x_train, y_train = data.get_train_data()
            x_train = tf.expand_dims(x_train, -1)
            x_train.set_shape([None,  config.input.img_out_shape[0],
                               config.input.img_out_shape[1],config.input.img_out_shape[2]])
            y_train.set_shape([None, config.input.mask_out_shape[0],
                               config.input.mask_out_shape[1], config.input.mask_out_shape[2]])
            y_train = tf.cast(y_train, tf.int32)

            batch_queue = [x_train, y_train]


        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        def clone_fn(batch_queue):
            x_train, y_train = batch_queue
            print(x_train)
            print(y_train)
            y_input = tf.expand_dims(tf.cast(y_train, tf.float32),-1)
            f_score, end_points, fc_fake, fc_true = net.net(x_train, y_input)
            # Add loss function.
            loss_G, loss_D = net.loss(f_score, y_train, fc_fake, fc_true)
            return f_score, end_points, x_train, y_train, loss_G, loss_D

        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        clones = deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

        for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
            summaries.add(tf.summary.scalar(loss.op.name, loss))

        f_score, _, x_train, y_train, loss_G, loss_D= clones[0].outputs
        #print(x_train)
        y_train_hot = tf.one_hot(y_train, depth=config.network.num_classes, axis=-1)
        ## add precision and recall
        f_score = tf.cast(tf.argmax(f_score, -1), tf.int32)
        f_score_hot = tf.one_hot(f_score, depth=config.network.num_classes, axis=-1)
        pred = tf.reduce_sum(f_score_hot * y_train_hot, axis=(0, 1, 2, 3))
        all_pred = tf.reduce_sum(f_score_hot, axis=(0, 1, 2, 3)) + 1e-5
        all_true = tf.reduce_sum(y_train_hot, axis=(0, 1, 2, 3)) + 1e-5
        recall = pred/all_pred
        prec = pred/all_true
        with tf.variable_scope('evaluation'):
            for i in range(config.network.num_classes):
                summaries.add(tf.summary.scalar('{}th_class_precision'.format(i), prec[i]))
                summaries.add(tf.summary.scalar('{}th_class_recall'.format(i), recall[i]))

        #################################
        # Configure the moving averages #
        #################################
        if config.train.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                config.train.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf_utils.configure_learning_rate(config,
                                                             global_step)
            optimizer = tf_utils.configure_optimizer(config.train, learning_rate)

        if config.train.moving_average_decay:
            update_ops.append(variable_averages.apply(moving_average_variables))


        op_g = optimizer.minimize(loss_G)
        op_d = optimizer.minimize(loss_D)

        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                           first_clone_scope))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config.deploy.gpu_memory_fraction)
        configproto = tf.ConfigProto(gpu_options=gpu_options,
                                     log_device_placement=False,
                                     allow_soft_placement=True, )

        saver = tf.train.Saver(max_to_keep=100)

        scaffold = tf.train.Scaffold(
                    init_op=None,
                    init_feed_dict=None,
                    init_fn=tf_utils.get_init_fn(config),
                    ready_op=None,
                    ready_for_local_init_op=None,
                    local_init_op=[data.get_iterator().initializer],
                    summary_op=summary_op,
                    saver=saver,
                    copy_from_scaffold=None
                    )

        ckpt_hook = tf.train.CheckpointSaverHook(
                    checkpoint_dir=config.summary.train_dir,
                    save_secs=config.summary.save_checkpoint_secs,
                    save_steps=config.summary.save_checkpoint_steps,
                    saver=None,
                    checkpoint_basename='model.ckpt',
                    scaffold=scaffold,
                    listeners=None
        )
        sum_writer = tf.summary.FileWriter(logdir=config.summary.train_dir)
        sum_hook = tf.train.SummarySaverHook(
                    save_steps=None,
                    save_secs=config.summary.save_summaries_secs,
                    output_dir=config.summary.train_dir,
                    summary_writer=sum_writer,
                    scaffold=None,
                    summary_op=summary_op,
        )

        with tf.train.MonitoredTrainingSession(
                master='',
                is_chief=True,
                checkpoint_dir=config.summary.train_dir,
                scaffold=scaffold,
                hooks=[ckpt_hook,sum_hook],
                save_checkpoint_secs=None,
                save_summaries_steps=None,
                save_summaries_secs=None,
                config=configproto,
                log_step_count_steps=config.summary.log_every_n_steps) as sess:
            while not sess.should_stop():

                _, loss_g, g_step = sess.run([op_g, loss_G, global_step])
                _, loss_d = sess.run([op_d, loss_D])
                print("{} step loss_g is {}, loss_d is {}".format(g_step, loss_g, loss_d))



if __name__ == '__main__':
    tf.app.run()
