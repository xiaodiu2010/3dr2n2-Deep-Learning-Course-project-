import tensorflow as tf
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from nets import sample_net
from nets import nets_factory
#from tensorlayer.cost import dice_coe
slim = tf.contrib.slim



class singleview(object):
    def __init__(self, config):
        self.config = config
        self.network = nets_factory.get_network_fn(self.config)
        if self.config.train.use_batch:
            self.normalizer_fn = slim.batch_norm
            self.batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': self.config.input.is_train
            }
            print("Using Batch Normalization")
        else:
            self.normalizer_fn = None
            self.batch_norm_params = None
            print("Not Using Batch Normalization")

    def net(self, images, y_input, reuse=False):
        '''
        with slim.arg_scope(net.arg_scope(weight_decay=self.config.train.weight_decay,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.batch_norm_params)):
        '''
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        logits, end_points = self.network(images, self.config, reuse=reuse)
        dims = end_points['global_pool'].get_shape().as_list()
        fc_feature_fake = tf.squeeze(end_points["global_pool"])
        fc_feature_fake.set_shape((None, dims[-1]))
        fc_feature_fake = slim.fully_connected(fc_feature_fake, 343)

        with tf.variable_scope('3dconv', reuse=False):
            with slim.arg_scope(self.arg_scopes_3dconv()):
                net = slim.conv3d(y_input, 16, 3, stride=1, padding='SAME')
                end_points['conv1'] = net

                net = slim.conv3d(net, 32, 3, stride=2, padding='SAME')
                end_points['conv2'] = net

                net = slim.conv3d(net, 64, 3, stride=1, padding='SAME')
                end_points['conv3'] = net

                net = slim.conv3d(net, 128, 3, stride=2, padding='SAME')
                end_points['conv3'] = net

                net = slim.conv3d(net, 128, 3, stride=1, padding='SAME')
                end_points['conv4'] = net

                net = slim.conv3d(net, 128, 3, stride=2, padding='SAME')
                end_points['conv5'] = net

                dims = net.get_shape().as_list()
                net = slim.flatten(net)
                net = slim.fully_connected(net, 343)
                end_points['fc1'] = net
                fc_feature_true = end_points['fc1']
                print(end_points)

        with tf.variable_scope('discriminator') as discriminator_scope:
            with slim.arg_scope(self.arg_scopes_discriminator()):
                ## Domain Transfer
                net_fake = slim.fully_connected(fc_feature_fake, 128, scope='fc1')
                net_fake = slim.fully_connected(net_fake, 64, scope='fc2')
                net_fake = slim.fully_connected(net_fake, 32, scope='fc3')
                net_fake = slim.fully_connected(net_fake, 16, scope='fc4')
                net_fake = slim.fully_connected(net_fake, 2, scope='fc5')
                discriminator_scope.reuse_variables()
                net_true = slim.fully_connected(fc_feature_true, 128, scope='fc1')
                net_true = slim.fully_connected(net_true, 64, scope='fc2')
                net_true = slim.fully_connected(net_true, 32, scope='fc3')
                net_true = slim.fully_connected(net_true, 16, scope='fc4')
                net_true = slim.fully_connected(net_true, 2, scope='fc5')


        ## 3D deconvlution
        with tf.variable_scope('3ddeconv', reuse=False):
            with slim.arg_scope(self.arg_scopes_3ddeconv()):
                net = slim.fully_connected(fc_feature_fake, dims[1]*dims[2]*dims[3]*dims[4])
                net = tf.reshape(net, [-1, dims[1], dims[2], dims[3],dims[4]])
                net = slim.conv3d_transpose(net, n_deconvfilter[1], 3, stride=[2,2,2])
                temp = net
                net = slim.conv3d(net, n_deconvfilter[1], 3)
                net = slim.conv3d(net, n_deconvfilter[1], 3)
                net = tf.concat([net, temp], -1)
                end_points['deconv1'] = net

                net = slim.conv3d_transpose(net, n_deconvfilter[2], 3, stride=[2,2,2])
                temp = net
                net = slim.conv3d(net, n_deconvfilter[2], 3)
                net = slim.conv3d(net, n_deconvfilter[2], 3)
                net = tf.concat([net, temp], -1)
                end_points['deconv2'] = net

                net = slim.conv3d_transpose(net, n_deconvfilter[3], 3, stride=[2,2,2])
                temp = net
                net = slim.conv3d(net, n_deconvfilter[3], 3)
                net = slim.conv3d(net, n_deconvfilter[3], 3)
                net = tf.concat([net, temp], -1)
                end_points['deconv3'] = net

                temp = net
                net = slim.conv3d(net, n_deconvfilter[4], 3)
                net = slim.conv3d(net, n_deconvfilter[4], 3)
                net = tf.concat([net, temp], -1)
                end_points['deconv4'] = net

                net = slim.conv3d(net, n_deconvfilter[4], 3)
                net = slim.conv3d(net, n_deconvfilter[4], 3, activation_fn=None)
                f_score = slim.conv3d(net, n_deconvfilter[5], 3, activation_fn=None)

        return f_score, end_points, net_fake, net_true


    def arg_scopes_discriminator(self):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer()) as sc:
                return sc


    def arg_scopes_3ddeconv(self):
        with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                            activation_fn=tf.nn.leaky_relu,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(self.config.train.weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv3d],
                                padding='SAME',
                                data_format=self.config.input.data_format) as sc:
                return sc

    def arg_scopes_3dconv(self):
        with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=self.normalizer_fn,
                            normalizer_params=self.batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(self.config.train.weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv3d],
                                padding='SAME',
                                data_format=self.config.input.data_format) as sc:
                return sc


    def loss(self, y_pred_cls, y_true_cls, fc_fake, fc_true, lam=0.01):
        loss_cls = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls, name="loss_cls")
        loss_cls = tf.reduce_mean(loss_cls)

        dims = fc_true.get_shape().as_list()
        loss_true = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fc_true, logits=np.ones(dims[0], tf.int32), name="loss_true")
        loss_fake = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fc_fake, logits=np.zeros(dims[0], tf.int32), name="loss_fake")
        loss_fake_reverse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=fc_fake, logits=np.ones(dims[0], tf.int32), name="loss_reverse")

        loss_G = loss_cls + lam * loss_fake_reverse
        loss_D = loss_fake + loss_true

        with tf.name_scope('total'):
            tf.add_to_collection('loss_cls', loss_cls)
            tf.add_to_collection('loss_true', loss_true)
            tf.add_to_collection('loss_fake', loss_fake)
            tf.add_to_collection('loss_reverse', loss_fake_reverse)
            tf.add_to_collection('loss_G', loss_G)
            tf.add_to_collection('loss_D', loss_D)
        tf.losses.add_loss(loss_G)
        tf.losses.add_loss(loss_D)
        return loss_G, loss_D



