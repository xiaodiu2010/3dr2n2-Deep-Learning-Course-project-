import os
import sys
import glob
import numpy as np
import cv2
import itertools
from .binvox_rw import *

def get_files(folder, seq_len=3 ,suffix='.png'):
    """
    Given path of the folder, returns list of files in it
    :param folder:
    :return:
    """
    folders = glob.glob(folder + "/*/*/*")
    print(len(folders))
    filenames = [get_seq(fold, seq_len, suffix) for fold in folders]
    print(len(filenames))
    filenames = list(itertools.chain.from_iterable(filenames))
    print("number of files : " + str(len(filenames)))
    return filenames


def get_seq(folder, seq_len=3, suffix='png'):
    images = glob.iglob(folder + "/*" + suffix)
    images = list(images)
    #images = glob.glob("../data/ShapeNetRendering/02691156/eb2fbd46563e23635fc197bbabcd5bd/rendering" + "/*.png")
    #print(len(images))
    num_seq = len(images) - (seq_len-1)
    return [[images[i] for i in np.arange(k, k + seq_len)] for k in range(num_seq)]


def get_labels(folder, filenames):
    labelnames = [folder +'/'+ '/'.join(file[0].split('/')[-4:-2]) + '/model.binvox' for file in filenames]
    return labelnames


def _read_py_function(filenames, label):
    images = [cv2.imread(filename.decode('utf-8')) for  filename in filenames]
    image_decoded = np.stack(images)
    #print(image_decoded.shape)
    with open(label, 'rb') as f:
        dims, translate, scale = read_header(f)
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool)
        mask = data.reshape(dims)
        #ask = np.transpose(mask, (0, 2, 1))

    return image_decoded.astype(np.uint8), mask.astype(np.uint8)


def _read_py_function_test(filenames):
    images = [cv2.imread(filename.decode('utf-8')) for  filename in filenames]
    image_decoded = np.stack(images)
    #print(image_decoded.shape)
    return image_decoded.astype(np.uint8)


def split_train_valid(folder):
    folders = glob.glob(folder + "*")
    print(folders)
    num_folders = len(folders)
    index = np.random.permutation(np.arange(num_folders))
    folders = np.array(folders)
    train_folders = folders[index[:int(0.9 * num_folders)]]
    valid_folders = folders[index[int(0.9 * num_folders):]]
    return train_folders, valid_folders