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
    batch_size = 4

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

    # init the model
    model = VQVAE(n_token=n_class, codebook_szie=512, codebook_dim=512, beta=0.1)
    model.cuda()
    model.train()

    # unpack
    train_x = train_data['x']           # shape : 1625*3584*7
    train_y = train_data['y']           # shape : 1625*3584*7
    num_batch = len(train_x) // batch_size
    
    print('    num_batch:', num_batch)
    print('    train_x:', train_x.shape)
    print('    train_y:', train_y.shape)

    for bidx in range(num_batch):
         # index
        bidx_st = batch_size * bidx
        bidx_ed = batch_size * (bidx + 1)

        # unpack batch data
        batch_x = train_x[bidx_st:bidx_ed]
        batch_y = train_y[bidx_st:bidx_ed]

        # to tensor
        batch_x = torch.from_numpy(batch_x).long().cuda()
        batch_y = torch.from_numpy(batch_y).long().cuda()
    
        output = model.encoder(batch_x)

        loss = output-batch_x





if __name__ == '__main__':
    train()