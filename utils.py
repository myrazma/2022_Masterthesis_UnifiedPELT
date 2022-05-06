# Copyright (c) Meta Platforms, Inc. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import os
import subprocess
import time
from datetime import datetime
import pandas as pd


def freeze_params_by_layers(model, num_enc_layers, num_frozen_layers=0):
    additional_frozen_params = ['model.shared.weight', 'model.encoder.embed_positions.weight',
                                'model.decoder.embed_positions.weight']
    for name, par in model.named_parameters():
        if name in additional_frozen_params:
            par.requires_grad = False
        elif not name.startswith('model'):
            print(f'{name} will update!')
        else:
            try:
                layer_idx = int(name.split('.')[3])
            except ValueError:
                par.requires_grad = False
                continue
            is_decoder = 'decoder' in name
            if is_decoder:
                layer_idx += num_enc_layers
            if layer_idx < num_frozen_layers:
                par.requires_grad = False


def freeze_params(model, except_para_l=()):
    for name, par in model.named_parameters():
        skip = False
        for except_para in except_para_l:
            if except_para in name:
                # print(f'{name} |skipped when alterning requires_grad')
                skip = True
                break
        if skip:
            continue
        par.requires_grad = False


def unfreeze_params(model, except_para=None):
    for name, par in model.named_parameters():
        if except_para is not None and except_para in name:
            par.requires_grad = False
        else:
            par.requires_grad = True


def get_gpu_memory_map():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free,utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_info = [eval(x) for x in result.strip().split('\n')]
    gpu_info = dict(zip(range(len(gpu_info)), gpu_info))
    sorted_gpu_info = sorted(gpu_info.items(), key=lambda kv: kv[1][0], reverse=True)
    sorted_gpu_info = sorted(sorted_gpu_info, key=lambda kv: kv[1][1])
    return sorted_gpu_info


def choose_gpu(n_gpus=1, min_gpu_memory=6000, retry=False, sleep_time=30):
    start_time = time.time()
    sorted_gpu_info = get_gpu_memory_map()
    try:
        gpustat = subprocess.check_output(
            [
                'gpustat'
            ], encoding='utf-8')
        print(gpustat)
    except Exception as e:
        print(e)
    print(f'gpu_id, (mem_left, util): {sorted_gpu_info}')
    while True:
        gpus = []
        for gpu_id, (mem_left, util) in sorted_gpu_info:
            if mem_left >= min_gpu_memory:
                gpus.append(gpu_id)
                print('use gpu:{} with {} MB left, util {}%'.format(gpu_id, mem_left, util))
            if len(gpus) == n_gpus:
                # print('max num of gpus reached.')
                break
        if len(gpus) == 0:
            if retry:
                print(f'[{datetime.now().strftime("%H:%M:%S")}'
                      f' waited {time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))}]'
                      f' no gpu has memory >= {min_gpu_memory} MB, sleep {sleep_time}s...', end='\r')
                time.sleep(sleep_time)
            else:
                print(f'no gpu has memory >= {min_gpu_memory} MB, exiting...')
                exit()
        else:
            break
        sorted_gpu_info = get_gpu_memory_map()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    visible_gpus = ','.join([str(gpu_id) for gpu_id in gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus



# ---------------------------- Extend with My Methods - (Myra Zmarsly)----------------------------
def load_data(train_file='', dev_file='', dev_label_file='', test_file=''):
    # dev and train data
    # TODO check if files are available / downloaded
    data_train = pd.read_csv(train_file, sep='\t') if train_file != '' else None
    features_dev = pd.read_csv(dev_file, sep='\t') if dev_file != '' else None
    labels_dev = pd.read_csv(dev_label_file, sep='\t', header=None) if dev_label_file != '' else None
    data_test = pd.read_csv(test_file, sep='\t') if test_file != '' else None
    
    if features_dev is not None:
        # specify label columns
        label_columns = ['empathy', 'distress', 'emotion', 'personality_conscientiousness', 'personality_openess', 'personality_extraversion', 'personality_agreeableness', 'personality_stability', 'iri_perspective_taking',  'iri_personal_distress', 'iri_fantasy','iri_empathatic_concern']
        # since dev labels initially have no column names, add them manually
        labels_dev.columns = label_columns
        data_dev = features_dev.join(labels_dev)
    return data_train, data_dev, data_test


def clean_raw_data(data_df):
    """Preprocess raw data and dev data including the following steps:
    - remove empathy_bin and distress_bin as they are not appearing in the 
    - remove iri labels
    - remove personality labels

    Args:
        data_df (_type_): _description_
        features_dev (_type_): _description_
        labels_dev (_type_): _description_
    """
    # clean data from unnecessary files
    # remove empathy and distress bin as we dont need it here
    cols_to_drop = ['empathy_bin', 'distress_bin']
    # create loop to check if the labels are in the data or not
    for col in cols_to_drop:
        if col in data_df.columns:
            data_df.drop([col], inplace=True, axis=1)

    # remove iri and personal things a labels here
    necessary_cols = [col for col in data_df.columns if not (col.__contains__('personality') or col.__contains__('iri'))]
    # additionally remove ids for now
    necessary_cols = [col for col in necessary_cols if not col.__contains__('id')]
    data_df = data_df[necessary_cols]
    return data_df

