from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf

from models import tdr2n2
from models import autoencoder
from models import svrecon

slim = tf.contrib.slim

networks_map = {'sv_reconstruction' : svrecon.singleview,
                'auto_encoder'  : autoencoder.autoencoder,
                'tdr2n2': tdr2n2.TDR2N2
               }



def get_model_fn(config):

  if config.network.model_name not in networks_map:
    raise ValueError('Name of model unknown %s' % config.network.model_name)
  func = networks_map[config.network.model_name]

  return func