import os, sys
sys.path.append('./')
sys.path.append('../')
import tensorflow as tf
import datetime
from utils.config import process_config
from data_loader.data_generator import DataGenerator

slim = tf.contrib.slim

global_step = tf.Variable(0, trainable=False)
config = '../configs/example.json'


config = process_config(config)
print(config)

dataset_train = DataGenerator(config.input)

x_train, y_train = dataset_train.get_train_data()


sess = tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=600,
        save_summaries_steps=100,
        save_summaries_secs=None,
        config=None,
        stop_grace_period_secs=120,
        log_step_count_steps=100
)

step = 0
sess.run(dataset_train.get_iterator().initializer)
while not sess.should_stop():
    start_time = datetime.datetime.now()
    g_step = sess.run([global_step])
    print(g_step)
    step += 1




