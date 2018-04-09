import tensorflow as tf
import os, sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
from nets import sample_net
from nets import nets_factory
#from tensorlayer.cost import dice_coe
slim = tf.contrib.slim



class TDR2N2(object):
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

    def net(self, images, reuse=False):
        '''
        with slim.arg_scope(net.arg_scope(weight_decay=self.config.train.weight_decay,
                                          normalizer_fn=self.normalizer_fn,
                                          normalizer_params=self.batch_norm_params)):
        '''
        n_deconvfilter = [128, 128, 128, 64, 32, 2]
        fc_features = []
        reuse=False
        end_points = None
        for i in range(self.config.input.seq_len):
            image = images[:, i, :, :, :]
            dims = images.get_shape().as_list()
            print(image)
            image = tf.reshape(image, [-1, dims[2], dims[3], dims[4]])
            logits, end_points = self.network(image, self.config, reuse=reuse)
            dims = end_points['global_pool'].get_shape().as_list()
            fc_feature = tf.squeeze(end_points["global_pool"])
            fc_feature.set_shape((None, dims[-1]))
            fc_features.append(fc_feature)
            if not reuse:
                end_points = end_points
                print("global_pool : {}".format(end_points['global_pool']))
            reuse = True
        print(fc_features)
        fc_features = tf.stack(fc_features)
        print("fc_features: {}".format(fc_features))
        dims = fc_features.get_shape().as_list()
        print("dims" + str(dims))

        with tf.variable_scope('3dlstm', reuse=False):
            with slim.arg_scope(self.arg_scopes_lstm()):
                ## 3d LSTM
                h = [None for i in range(self.config.input.seq_len+1)]
                h[0] = tf.zeros((self.config.input.batch_size, 4, 4, 4, n_deconvfilter[0]))
                for i in range(self.config.input.seq_len):
                    fc_feature = fc_features[i]
                    h[i+1] = self.tdgru(h[i], fc_feature, n_deconvfilter[0])

                print(h[-1])
        ## 3D deconvlution
        with tf.variable_scope('3ddeconv', reuse=False):
            with slim.arg_scope(self.arg_scopes_3ddeconv()):
                net = slim.conv3d_transpose(h[-1], n_deconvfilter[1], 3, stride=[2,2,2])
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

                net = slim.conv3d(net, n_deconvfilter[4], 3)
                temp = net
                net = slim.conv3d(net, n_deconvfilter[4], 3)
                net = slim.conv3d(net, n_deconvfilter[4], 3)
                net = tf.concat([net, temp], -1)
                end_points['deconv4'] = net

                f_score = slim.conv3d(net, n_deconvfilter[5], 3)

        return f_score, end_points


    def tdgru(self, h_prev, fc_feature, filters):
        u_t = tf.sigmoid(self.fcconv3dlayer(h_prev, fc_feature, filters))
        r_t = tf.sigmoid(self.fcconv3dlayer(h_prev, fc_feature, filters))
        h_t = tf.multiply((1. - u_t), h_prev) + \
              tf.multiply(u_t, tf.tanh(self.fcconv3dlayer(tf.multiply(r_t, h_prev), fc_feature, filters)))
        return h_t

    def fcconv3dlayer(self, h_prev, fc_feature, filters):
        out_shape = h_prev.get_shape().as_list()
        fc_output = tf.reshape(slim.fully_connected(fc_feature, 4*4*4*filters), out_shape)
        h_next = fc_output + slim.conv3d(h_prev, filters, [3,3,3])
        return h_next

    def arg_scopes_lstm(self):
        with slim.arg_scope([slim.conv3d, slim.fully_connected],
                            activation_fn=None,
                            normalizer_fn=None,
                            normalizer_params=None,
                            weights_regularizer=None,
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv3d],
                                padding='SAME',
                                data_format=self.config.input.data_format) as sc:
                return sc


    def arg_scopes_3ddeconv(self):
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

    def loss(self, y_pred_cls, y_true_cls):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true_cls, logits=y_pred_cls, name="loss")
        loss = tf.reduce_mean(loss)
        with tf.name_scope('total'):
            tf.add_to_collection('EXTRA_LOSSES', loss)
        tf.losses.add_loss(loss)
        return loss



