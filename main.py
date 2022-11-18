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

import Saver


path_data_root = '/home/zhangzhedong/cp-word-transformer/dataset/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_test_data = os.path.join(path_data_root, 'test_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')
path_exp = 'exp_log'

gid = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)

def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def train():
    n_epoch = 4000
    max_grad_norm = 3
    batch_size = 32

    # load the dictionary of every token type
    # tempo,chord,bar-beat,type,pitch,duration,velocity
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)
    test_data = np.load(path_test_data)

    # create the saver
    saver_agent = Saver.Saver(path_exp)

    # config
    n_class = []
    for key in event2word.keys():
        n_class.append(len(dictionary[0][key]))

    # log
    print('num of classes:', n_class)

    # init the model
    model = VQVAE(batch_size=batch_size, n_token=n_class, codebook_szie=512, codebook_dim=512, beta=0.1)
    model.cuda()
    model.train()

    n_parameters = network_paras(model)
    print('n_parameters: {:,}'.format(n_parameters))
    saver_agent.add_summary_msg(
        ' > params amount: {:,d}'.format(n_parameters))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # unpack
    train_x = train_data['x']           # shape : 1625*3584*7
    num_batch = len(train_x) // batch_size

    test_x = test_data['x']
    num_test_batch = len(test_x) // batch_size
    
    print('    num_batch:', num_batch)
    print('    train_x:', train_x.shape)

    strat_time = time.time()
    for epoch in range(n_epoch):
        epoch_rec_loss = 0
        epoch_vq_loss = 0
        epoch_loss = 0
        model.train()

        for bidx in range(num_batch):
            # index
            bidx_st = batch_size * bidx
            bidx_ed = batch_size * (bidx + 1)

            # unpack batch data
            batch_x = train_x[bidx_st:bidx_ed]

            # to tensor
            batch_x = torch.from_numpy(batch_x).long().cuda()

            # run the model
            decoder_output, encoder_input, loss_rec, vq_loss = model(batch_x)
            losses = loss_rec + vq_loss

            # update
            model.zero_grad()
            losses.backward()
            optimizer.step()

            # print
            sys.stdout.write('{}/{} | Rec_Loss: {:06f} | VQ_Loss: {:06f} | Total_loss: {:06f} \r'.format(
                bidx, num_batch, loss_rec, vq_loss, losses))
            sys.stdout.flush()

            # epoch loss
            epoch_rec_loss += loss_rec
            epoch_vq_loss += vq_loss
            epoch_loss += losses

        run_time = time.time() - strat_time
        epoch_rec_loss = epoch_rec_loss / num_batch
        epoch_vq_loss = epoch_vq_loss / num_batch
        epoch_loss = epoch_loss / num_batch

        print('\r ------------------------------------------------------------------------ \r')
        print('epoch: {}/{} | REC_LOSS: {:06f} | VQ_LOSS: {:06f} | Loss: {:06f} | time: {}'.format(
            epoch, n_epoch, epoch_rec_loss, epoch_vq_loss, epoch_loss, str(datetime.timedelta(seconds=run_time))))
        
        saver_agent.add_summary('epoch loss', epoch_loss)
        saver_agent.add_summary('epoch rec loss', epoch_rec_loss)
        saver_agent.add_summary('epoch vq loss', epoch_vq_loss)
        saver_agent.add_summary('time cost', str(datetime.timedelta(seconds=run_time)))



        # if (epoch) % 20 == 0:
        #     test_epoch_loss = 0
        #     model.eval()
        #     for bidx in range(num_test_batch):
        #         # index
        #         bidx_st = batch_size * bidx
        #         bidx_ed = batch_size * (bidx + 1)

        #         # unpack batch data
        #         batch_x = test_x[bidx_st:bidx_ed]

        #         # to tensor
        #         batch_x = torch.from_numpy(batch_x).long().cuda()

        #         # run the model
        #         decoder_output, encoder_input, loss_rec, vq_loss = model(batch_x)
        #         losses = loss_rec + vq_loss
        #         test_epoch_loss += losses

        #         # print
        #         sys.stdout.write('TEST_epoch: {}/{} | Rec_Loss: {:06f} | VQ_Loss: {:06f} | Total_loss: {:06f} \r'.format(
        #         bidx, num_test_batch, loss_rec, vq_loss, losses))
        #         sys.stdout.flush()
                

        #     run_time = time.time() - strat_time
        #     epoch_loss = test_epoch_loss / num_batch
        #     print('------------------------------------')
        #     print('Test epoch: {}/{} | Loss: {} | time: {}'.format(
        #     epoch/20, n_epoch/20, epoch_loss, str(datetime.timedelta(seconds=run_time))))

        # save model, with policy
        loss = epoch_loss
        if 0.5 < loss <= 1.0:
            fn = int(loss * 10) * 10
            saver_agent.save_model(model, name='loss_' + str(fn))
        elif 0.05 < loss <= 0.5:
            fn = int(loss * 100)
            saver_agent.save_model(model, name='loss_' + str(fn))
        elif loss <= 0.05:
            print('Finished')
            return  
        else:
            saver_agent.save_model(model, name='loss_high')
            


            

        

        
        

if __name__ == '__main__':

    train()