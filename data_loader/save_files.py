import os, sys
sys.path.append('./')
sys.path.append('../')
import pickle
from data_loader.util import *
from utils.config import process_config


if __name__ == '__main__':
    config = '../configs/example.json'

    config = process_config(config)
    print(config)


    #files = get_files(config.input.train_file_path, seq_len=config.input.seq_len, suffix='.png')
    files = get_files_3d(config.input.train_file_path)
    with open("../data/files.txt", "wb") as fp:
        pickle.dump(files, fp)


    with open("../data/files.txt", "rb") as fp:
        test = pickle.load(fp)
        print(len(test))