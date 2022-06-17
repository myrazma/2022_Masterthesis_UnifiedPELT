# Copyright (c) Meta Platforms, Inc. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

import numpy as np
import os
import subprocess
import time
from datetime import datetime
import pandas as pd
from lora_layers import LoRA_Linear


from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from .transformers.adapters.layer import AdapterLayerBaseMixin
from .transformers.models.bert.modeling_bert import BertSelfAttention


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


def get_gates(model, epoch=None, split=None):
    """Store the gates from the model, possible to also store the epoch and splits trained on

    Args:
        model (_type_): _description_
    """
    # check if gates do exist:
    # are adapters, lora or prefix training used? if not, don't store them
    
    # Make table as follows:
    # model_name | gate_lora | gate_prefix | gate_adapters | encoder_layer | epoch | split 

    # str | float or None | float or None | float or None | int or None | str or None
    
    gate_output_d, lora_gate_query, lora_gate_value, prefix_gate = np.nan, np.nan, np.nan, np.nan
    for idx, layer in enumerate(model.bert.encoder.layer):
        # check the variables for gating in each bert encoder layer
        if layer.output.adapters: # if not empty adapters ModuleDict
            try:
                #gate_output_d = np.array(layer.output.gate_output_d).reshape((-1,))
                gate_output_d = [gate for batch in layer.output.gate_output_d for gate in list(batch)]
            except:
                print('len(layer.output.gate_output_d) did not work')
                    
        if isinstance(layer.attention.self.query, LoRA_Linear):
            lora_gate_query = [gate for batch in layer.attention.self.query.lora_gate_output_l for gate in list(batch)]
        
        if isinstance(layer.attention.self.value, LoRA_Linear):
            #print('layer.attention.self.value.lora_gate_output_l:', len(layer.attention.self.value.lora_gate_output_l))
            #lora_gate_value = np.array(layer.attention.self.value.lora_gate_output_l).reshape(-1)
            lora_gate_value = [gate for batch in layer.attention.self.value.lora_gate_output_l for gate in list(batch)]
        
        if isinstance(layer.attention.self, BertSelfAttention):
            #print('layer.attention.self.prefix_gate_output_l:', len(layer.attention.self.prefix_gate_output_l))
            #prefix_gate = np.array(layer.attention.self.prefix_gate_output_l).reshape(-1)
            prefix_gate = [gate for batch in layer.attention.self.prefix_gate_output_l for gate in list(batch)]

        gate_dict = pd.DataFrame({'gate_prefix': prefix_gate, 'gate_lora_value': lora_gate_value, 'gate_lora_query': lora_gate_query, 'gate_adapters': gate_output_d, 'encoder_layer': idx, 'epoch':epoch, 'split': split})

        return gate_dict