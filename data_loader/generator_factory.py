from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from data_loader.data_generator_3d import  DataGenerator_3d
from data_loader.data_generator_lstm import  DataGenerator_lstm

slim = tf.contrib.slim

generators_map = {'generator_3d' : DataGenerator_3d,
                'generator_lstm'  : DataGenerator_lstm
               }



def get_generator_fn(config):

  if config.network.generator not in generators_map:
    raise ValueError('Name of generator unknown %s' % config.network.generator)
  func = generators_map[config.network.generator]

  return func