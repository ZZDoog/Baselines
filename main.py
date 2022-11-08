import sys
import os


import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from models import VQVAE, VectorQuantizer
from gen_midi import write_midi


path_data_root = '/home/zhangzhedong/cp-word-transformer/dataset/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')
path_exp = 'exp'


def train():
    n_epoch = 4000
    max_grad_norm = 3

    # load the dictionary of every token type
    # tempo,chord,bar-beat,type,pitch,duration,velocity
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)


if __name__ == '__main__':
    train()