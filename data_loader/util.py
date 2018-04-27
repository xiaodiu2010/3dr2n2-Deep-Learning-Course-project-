import matplotlib

matplotlib.use('Agg')
import os
import sys
import glob
import numpy as np
import cv2
import itertools
from .binvox_rw import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import string
import random

def get_files_3d(folder):
    folders = glob.glob(folder + "/*/*/*")
    print(len(folders))
    filenames = [list(glob.iglob(fold + '/*' + 'png')) for fold in folders]
    filenames = list(itertools.chain.from_iterable(filenames))
    print(filenames[0])
    print("number of files : " + str(len(filenames)))
    return filenames


def get_labels_3d(folder, filenames):
    labelnames = [folder +'/'+ '/'.join(file.split('/')[-4:-2]) + '/model.binvox' for file in filenames]
    print(labelnames[0])
    return labelnames

def _read_py_function_3d(filename, label):
    image = cv2.imread(filename.decode('utf-8'))
    with open(label.decode("utf-8"), 'rb') as f:
        dims, translate, scale = read_header(f)
        raw_data = np.frombuffer(f.read(), dtype=np.uint8)
        values, counts = raw_data[::2], raw_data[1::2]
        data = np.repeat(values, counts).astype(np.bool)
        mask = data.reshape(dims)
        #ask = np.transpose(mask, (0, 2, 1))

    return image.astype(np.uint8), mask.astype(np.uint8)


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
    with open(label.decode("utf-8"), 'rb') as f:
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

def save_result(path, x_train, y_train, f_score):
    ## create directory
    path = path + '/imgs/'
    if not os.path.isdir(path):
        os.mkdir(path)
    num = 4
    num_col = 3
    fig = plt.figure(figsize=(num*8, 18))
    print(x_train.shape)
    for i in range(num):
        img, mask, p_mask = x_train[i][0], y_train[i], f_score[i]
        img = (img/2. + 0.5) * 255.
        img = np.reshape(img, (137,137,3))
        fig.add_subplot(num, num_col, num_col * i + 1)
        plt.imshow(img.astype(np.uint8))
        axs = fig.add_subplot(num, num_col, num_col * i + 2, projection='3d')
        xs, ys, zs = mask.nonzero()
        axs.scatter(xs, ys, zs, c='red')
        axs = fig.add_subplot(num, num_col, num_col * i + 3, projection='3d')
        xs, ys, zs = p_mask.nonzero()
        axs.scatter(xs, ys, zs, c='blue')
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
    fig.savefig(path + name + ".png")