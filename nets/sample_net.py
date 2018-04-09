import tensorflow as tf

slim = tf.contrib.slim


def arg_scope(weight_decay=0.0005, normalizer_fn=None, normalizer_params=None,
              data_format="NHWC"):
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=normalizer_params,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        #weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            data_format=data_format) as arg_sc:
            return arg_sc


def encoder(inputs,
            reuse=False,
            scope='encoder'):
    with tf.variable_scope(scope, 'encoder', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d]):
            end_points = {}
            net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1')
            end_points['conv1'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool1')

            net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2')
            end_points['conv2'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool2')

            net = slim.repeat(net, 2, slim.conv2d, 256, 3, scope='conv3')
            end_points['conv3'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool3')

            net = slim.repeat(net, 2, slim.conv2d, 512, 3, scope='conv4')
            end_points['conv4'] = net
            net = slim.max_pool2d(net, [2, 2], scope='pool4')

            net = slim.repeat(net, 2, slim.conv2d, 1024, 3, scope='conv5')
            end_points['conv5'] = net

            return net, end_points
