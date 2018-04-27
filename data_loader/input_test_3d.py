import os, sys
sys.path.append('./')
sys.path.append('../')
import tensorflow as tf
import datetime
from utils.config import process_config
from data_loader.generator_factory import get_generator_fn
import matplotlib.pyplot as plt
import numpy as np
from data_loader.util import save_result
from mpl_toolkits.mplot3d import Axes3D

config = '../configs/svrecon.json'

config = process_config(config)
print(config)


dataset = get_generator_fn(config)(config.input)

x_train, y_train = dataset.get_train_data()
x_train.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
y_train.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1], config.input.mask_out_shape[2]])

x_test, y_test = dataset.get_eval_data()
x_test.set_shape([None, config.input.img_out_shape[0], config.input.img_out_shape[1], config.input.img_out_shape[2]])
y_test.set_shape([None, config.input.mask_out_shape[0], config.input.mask_out_shape[1], config.input.mask_out_shape[2]])

#print(x_train)

scaffold = tf.train.Scaffold(
    init_op=None,
    init_feed_dict=None,
    init_fn=None,
    ready_op=None,
    ready_for_local_init_op=None,
    local_init_op=[dataset.get_iterator().initializer,
                   dataset.get_iterator(is_train=False).initializer],
    summary_op=None,
    saver=None,
    copy_from_scaffold=None
)

sess = tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=scaffold,
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
visual = True
while not sess.should_stop():
    start_time = datetime.datetime.now()
    x, y, x_t, y_t= sess.run([x_train, y_train, x_test, y_test])
    step += 1

    print("max:{} min:{}".format(x.max(), x.min()))
    print(x.shape)
    print(np.unique(y))
    num = 4
    if visual:
        # Visualization
        if step % 10 == 0:
            fig= plt.figure(figsize=(num*4, 16))
            for i in range(num):
                img, mask, t_img, t_mask = x[i], y[i], x_t[i], y_t[i]
                fig.add_subplot(num, 4, 4*i+1)
                plt.imshow(img.astype(np.uint8))
                axs = fig.add_subplot(num, 4, 4*i+2, projection='3d')
                xs,ys,zs = mask.nonzero()
                axs.scatter(xs, ys, zs, c='red')
                fig.add_subplot(num, 4, 4*i+3)
                plt.imshow(t_img.astype(np.uint8))
                axs =fig.add_subplot(num, 4, 4*i+4, projection='3d')
                xs, ys, zs = t_mask.nonzero()
                axs.scatter(xs, ys, zs, c='blue')
            plt.show()

        #save_result('../data/', x, y, y)
