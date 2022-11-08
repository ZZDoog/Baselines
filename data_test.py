import os
import pickle
import json
import numpy as np

path_data_root = '/home/zhangzhedong/cp-word-transformer/dataset/representations/uncond/cp/ailab17k_from-scratch_cp'
path_train_data = os.path.join(path_data_root, 'train_data_linear.npz')
path_dictionary =  os.path.join(path_data_root, 'dictionary.pkl')

if __name__ == '__main__':

    dictionary = pickle.load(open(path_dictionary, 'rb'))
    event2word, word2event = dictionary
    train_data = np.load(path_train_data)
    class_keys = word2event.keys()

    train_x = train_data['x']           # shape : 1625*3584*7
    train_y = train_data['y']           # shape : 1525*3584*7
    train_mask = train_data['mask']     # shape : 1625*3584
    
    bar_cnt = 0
    sample_cnt = 0

    for sample in train_x:
        for i in range(len(sample)):
            vals = []
            for kidx, key in enumerate(class_keys):
                vals.append(word2event[key][sample[i][kidx]])
            
            if vals[3] == 'Metrical':
                if vals[2] == 'Bar':
                    bar_cnt += 1
        sample_cnt += 1
    
    # total num of bar: 168470
    # total num of sample: 1625
    # avarge: about 103 bar per song
    print('the total num of bar is:', bar_cnt)
    print('the total num of sample is:', sample_cnt)
    print('the avarge bar number in every song is:', bar_cnt/sample_cnt)    
    print(train_x.shape)