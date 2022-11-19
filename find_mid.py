import os
import sys

import numpy as np
import utils
import miditoolkit
from miditok import REMI, get_midi_programs
from miditoolkit import MidiFile



# 递归查找lmd_match文件夹中所有的midi文件
def get_all_midi2(path, dirlist=[], filelist=[]):
    flist = os.listdir(path)
    for filename in flist:
        subpath = os.path.join(path, filename)
        if os.path.isdir(subpath):
            dirlist.append(subpath)  # 如果是文件夹，添加到文件夹列表中
            get_all_midi2(subpath, dirlist, filelist)  # 向子文件内递归
        if os.path.isfile(subpath):
            filelist.append(subpath)  # 如果是文件，添加到文件列表中
    return dirlist, filelist




def midi2remi(file_path):

    note_items, tempo_items = utils.read_items(file_path)
    note_items = utils.quantize_items(note_items)
    chord_items = utils.extract_chords(note_items)
    items = chord_items + tempo_items + note_items
    max_time = note_items[-1].end
    groups = utils.group_items(items, max_time)
    events = utils.item2event(groups)

    return events


if __name__ == '__main__':

    # load the midi list
    midi_list = np.load('Giant_piano_midi_list.npy')

    # path = '/home/zhangzhedong/Giant_midi_piano'
    # dirlist, midi_list = get_all_midi2(path)

    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
    additional_tokens = {'Chord': True, 'Rest': True, 'Tempo': True, 'Program': False, 'TimeSignature': False,
                        'rest_range': (2, 8),  # (half, 8 beats)
                        'nb_tempos': 32,  # nb of tempo bins
                        'tempo_range': (40, 250)}  # (min, max)

    tokenizer = REMI(pitch_range, beat_res, nb_velocities, additional_tokens, mask=True)

    sum_token = 0
    max_token = 0

    # average token:14824.316173, max token:472507
    for i in range(len(midi_list)):
        midi = MidiFile(midi_list[i])
        tokens = tokenizer.midi_to_tokens(midi)
        if len(tokens) != 1:
            print(i)
            print(midi_list[i])
        
        else:
            sum_token += len(tokens[0])
            if len(tokens[0]) > max_token:
                max_token = len(tokens[0])

        sys.stdout.write('{}/{}, average token:{:02f}, max token:{} \r'.format(i+1, len(midi_list), sum_token/(i+1), max_token))
        sys.stdout.flush
    
    

    print('done')

