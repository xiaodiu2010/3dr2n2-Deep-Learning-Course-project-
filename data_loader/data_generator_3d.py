import os, sys
sys.path.append('./')
sys.path.append(os.path.dirname(__file__))
print(sys.path)
import tensorflow as tf
from .augmentor_3d import augmentation
from .util import *
import pickle
from . import util
import random

class DataGenerator_3d(object):
    def __init__(self, config):
        """
        config:
        data_paralle_size
        buffer:
        batch_size:
        """
        self.suffix = '.png'
        self.config = config
        #self.files_list = get_files_3d(config.train_file_path)
        with open("../data/files.txt", "rb") as fp:
            self.files_list = pickle.load(fp)
        self.train_filenames = self.files_list[:int(0.95*len(self.files_list))]
        random.shuffle(self.train_filenames)
        self.train_labels = get_labels_3d(config.train_label_path, self.train_filenames)
        self.eval_filenames = self.files_list[int(0.95*len(self.files_list)):]
        self.eval_labels = get_labels_3d(config.eval_label_path, self.eval_filenames)
        self.dataset = None
        self.test_dateset = None
        self.print_info()
        self.build_dataset()
        self.iterator_train = None
        self.iterator_test = None

    def print_info(self):
        print("The dataset has {} examples".format(len(self.files_list)))
        print("Their are {} images, and {} masks".format(len(self.train_filenames), len(self.train_labels)))
        print("Their are {} images, and {} masks in test set".format(len(self.eval_filenames), len(self.eval_labels)))

    def build_dataset(self):
        self.dataset = tf.data.Dataset.from_tensor_slices((self.train_filenames, self.train_labels))
        self.eval_dateset = tf.data.Dataset.from_tensor_slices((self.eval_filenames, self.eval_labels))

    def get_train_data(self):
        dataset_train = self.dataset.map(
                        lambda filename, label: tuple(tf.py_func(
                            self.load_data, [filename, label], [tf.uint8, tf.uint8])),
                        num_parallel_calls = self.config.data_parallel_threads)

        dataset_train = dataset_train.prefetch(self.config.batch_size)
        dataset_train = dataset_train.shuffle(self.config.buffer)
        dataset_train = dataset_train.repeat(1000)
        dataset_train = dataset_train.map(
                        lambda filename, label: tuple(tf.py_func(
                            augmentation(self.config), [filename, label], [tf.float32, tf.uint8])),
                        num_parallel_calls=self.config.data_parallel_threads)
        dataset_train = dataset_train.batch(self.config.batch_size)
        dataset_train = dataset_train.prefetch(4)

        self.iterator_train = dataset_train.make_initializable_iterator()
        X, y = self.iterator_train.get_next()
        return X, y


    def get_eval_data(self):
        dataset_eval = self.eval_dateset.map(
                        lambda filename, label: tuple(tf.py_func(
                            self.load_data, [filename, label], [tf.uint8, tf.uint8])),
                        num_parallel_calls = self.config.data_parallel_threads)
        dataset_eval = dataset_eval.prefetch(self.config.batch_size_test)
        dataset_eval = dataset_eval.repeat(1)
        dataset_eval = dataset_eval.map(
                        lambda filename, label: tuple(tf.py_func(
                            augmentation(self.config, is_train=False), [filename, label], [tf.float32, tf.uint8])),
                        num_parallel_calls=self.config.data_parallel_threads)
        dataset_eval = dataset_eval.batch(self.config.batch_size_test)
        dataset_eval = dataset_eval.prefetch(4)

        self.iterator_eval = dataset_eval.make_initializable_iterator()
        X_eval, y_eval = self.iterator_eval.get_next()
        return X_eval, y_eval


    def get_iterator(self, is_train=True):
        if is_train:
            return self.iterator_train
        else:
            return self.iterator_eval


    def load_data(self, filename, label):
        image, mask = util._read_py_function_3d(filename, label)
        return image, mask


