#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
# Copyright (c) Meta Platforms, Inc. All rights reserved.
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.

""" This is a copy of the run_glue.py file adapted for empathy prediction!

Adapting the glue classification tasks-file from UniPELT (https://github.com/morningmoni/UniPELT) to our task and data. 
For the not adapted, original version of this script from UniPELT, please have a look at the file run_glue.py.
Finetuning the library models for sequence regression on empathy and distress dataset from Buechel.

Original Authors: Mao, Yuning and Mathias, Lambert and Hou, Rui and Almahairi, Amjad and Ma, Hao and Han, Jiawei and Yih, Wen-tau and Khabsa, Madian
Paper: UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning
Year: 2022"""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
from os import path
import pickle
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from matplotlib import colors

sys.path.append('../')

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AdapterConfig,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    MultiLingAdapterArguments,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.adapters.composition import Stack
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
from utils import freeze_params, choose_gpu, freeze_params_by_layers, load_data, clean_raw_data
import preprocessing

# my imports
import torch
from torch.utils.tensorboard import SummaryWriter

from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import re

from lora_layers import LoRA_Linear
from transformers.adapters.layer import AdapterLayerBaseMixin
from transformers.models.bert.modeling_bert import BertSelfAttention

import pandas as pd
import importlib.util

from arguments import DataTrainingArguments, ModelArguments
from arguments import task_to_keys

# Setup wandb
package_name = 'wandb'
WANDB_AVAILABLE = importlib.util.find_spec("wandb") is not None
if WANDB_AVAILABLE:
    import wandb
else:
    print(package_name +" is not installed. Not used here.")



check_min_version("4.5.0")


logger = logging.getLogger(__name__)

#COLORS = ['#029e72', '#e69f00', '#f0e441', '#57b4e8']
COLORS = ['#D81B60', '#1E88E5', '#FFC107','#004D40','#029e72', '#e69f00', '#f0e441', '#57b4e8', '#b4a7d6', '#b1bab2']


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MultiLingAdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set run dir for Model
    training_args.logging_dir = data_args.tensorboard_output_dir  # write into the same directory

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Setup writer for tensorboard if --use_tensorboard is True
    # per default stores in runs/ + output_dir, where the output dir is set to '' per default
    tensorboard_writer = SummaryWriter(data_args.tensorboard_output_dir) if data_args.use_tensorboard else None

    # Added by Myra Z.
    # setup wandb, use wandb if available
    # if entity is empty or None, don't use wandb no matter if the package is available or not
    use_wandb = WANDB_AVAILABLE and (data_args.wandb_entity is not None or data_args.wandb_entity != '' or data_args.wandb_entity != 'None')
    print('Use wandb:', use_wandb)
    if use_wandb:  # should already be imported
        try:
            os.system('cmd /k "wandb login"')  # login
            wandb.init(project=data_args.wandb_project, entity=data_args.wandb_entity, name=data_args.tensorboard_output_dir[5:])
        except Exception as e:
            print(f'\n run model without wandb: \n {e} \n')
            
    # store model config
    if use_wandb:
        wandb.config.update({
            "train_adapter": adapter_args.train_adapter,
            "add_enc_prefix": model_args.add_enc_prefix,
            "add_lora": model_args.add_lora,
            "tune_bias": model_args.tune_bias,
            "learning_rate":training_args.learning_rate,
            "tensorboard_output_dir":data_args.tensorboard_output_dir,
            "max_epochs":training_args.num_train_epochs
            })

    # ------------------------------
    # Data Loading
    # edited by Myra Z.
    data_train_pd, data_dev_pd, data_test_pd = load_data(train_file=data_args.train_file, dev_file=data_args.validation_file, dev_label_file=data_args.validation_labels_file, test_file=data_args.test_file)
    data_train_pd = clean_raw_data(data_train_pd)
    data_dev_pd = clean_raw_data(data_dev_pd)
    data_test_pd = clean_raw_data(data_test_pd)

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    # load tokenizer an dpreprocess data
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    dataset_emp_train, dataset_dis_train = preprocessing.get_preprocessed_dataset(data_train_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['message_id'])
    dataset_emp_dev, dataset_dis_dev = preprocessing.get_preprocessed_dataset(data_dev_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['message_id'])
    dataset_emp_test, dataset_dis_test = preprocessing.get_preprocessed_dataset(data_test_pd, tokenizer, training_args.seed, return_huggingface_ds=True, padding=padding, additional_cols=['message_id'])
    
    # --- choose dataset and data loader based on empathy ---
    # per default use empathy label
    train_dataset = dataset_emp_train
    eval_dataset = dataset_emp_dev
    test_dataset = dataset_emp_test
    display_text = 'Using empathy data'
    if data_args.task_name == 'distress':
        train_dataset = dataset_dis_train  # needed for k fold
        eval_dataset = dataset_dis_dev  # needed for k fold
        test_dataset = dataset_dis_test
        display_text = "Using distress data"
    print('\n------------ ' + display_text + ' ------------\n')

    # Task selection was here before, but since we are only using one task (regression),
    # these settings can stay the same for us
    is_regression = True  
    num_labels = 1
    label_list = None  # edit this, if classification should be used

    # Load pretrained model 
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # additional arguments
    config.add_enc_prefix = model_args.add_enc_prefix
    config.add_dec_prefix = model_args.add_dec_prefix
    config.add_cross_prefix = model_args.add_cross_prefix
    config.prefix_len = model_args.prefix_len
    config.mid_dim = model_args.mid_dim
    if 'bert' in model_args.model_name_or_path:
        num_layers = config.num_hidden_layers
    elif 'bart' in model_args.model_name_or_path:
        num_layers = config.encoder_layers
    config.add_adapter_gate = model_args.add_adapter_gate
    config.add_prefix_gate = model_args.add_prefix_gate
    config.tune_bias = model_args.tune_bias
    config.add_lora = model_args.add_lora
    config.lora_r = model_args.lora_r
    config.lora_alpha = model_args.lora_alpha
    config.add_lora_gate = model_args.add_lora_gate
    config.add_central_gate = model_args.add_central_gate
    config.early_stopping_patience = data_args.early_stopping_patience

    if model_args.drop_first_layers == 0:
        config.drop_first_prefix_layers_enc = list(range(model_args.drop_first_prefix_layers_enc))
        config.drop_first_prefix_layers_dec = list(range(model_args.drop_first_prefix_layers_dec))
        config.drop_first_prefix_layers_cross = list(range(model_args.drop_first_prefix_layers_cross))
    else:
        # override by drop_first_layers
        model_args.drop_first_adapter_layers = model_args.drop_first_layers
        config.drop_first_prefix_layers_enc = list(range(model_args.drop_first_layers))
        config.drop_first_prefix_layers_dec = list(range(model_args.drop_first_layers - num_layers))
        config.drop_first_prefix_layers_cross = list(range(model_args.drop_first_layers - num_layers))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Added by Myra Z.
    # TODO: If this is working we should do this as a model parameter
    # we could also think about in stead of providing a boolean, 
    # we could provide a path to the adapter that we want to load
    #model_args.stacking_adapter  # the pathe to the stacking adapter
    #model_args.use_stacking_adapter  # If we should use the stacking adapter
    #model_args.train_all_gates_adapters  # If True, then all gate parameters fo the adapter will be set to trainable
    
    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.task_name
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                adapter_args.adapter_config,
                non_linearity=adapter_args.adapter_non_linearity,
                reduction_factor=adapter_args.adapter_reduction_factor,
                leave_out=list(range(model_args.drop_first_adapter_layers))
            )
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            # otherwise, add a fresh adapter
            else:
                model.add_adapter(task_name, config=adapter_config)

        if model_args.use_stacking_adapter:  # Added by Myra Z.
            # TODO: Make this adapter based on the model inputs
            # TODO: Might need to use the code from above
            try: # TODO: other adapter here

                emotion_adapter_name_path = data_args.data_dir + model_args.stacking_adapter
                emotion_adapter_name = model.load_adapter(emotion_adapter_name_path, load_as='emotion')
                #print('emotion_adapter_name:', emotion_adapter_name)
                #emotion_adapter_name = model.load_adapter(model_args.stacking_adapter, source="hf")
                #config = AdapterConfig.load("pfeiffer")
                #emotion_adapter_name = model.load_adapter("sentiment/imdb@ukp", config=config)
                #emotion_adapter_name = model.load_adapter(model_args.stacking_adapter, config=config)
            except Exception as e:
                print(f'\n Stacking adapter cannot be used adn is not being used. Exception: \n {e}')
                emotion_adapter_name = None
        else:
            emotion_adapter_name = None

        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None

        # Freeze all model weights except of those of this adapter
        model.train_adapter([task_name])
        # Set the adapters to be used in every forward pass
        # added by Myra Z.:
        # Added more adapter possibilities here
        active_adapters_list = [task_name]
        if lang_adapter_name: active_adapters_list.append(lang_adapter_name)
        if emotion_adapter_name: active_adapters_list.append(emotion_adapter_name)
        
        if model_args.use_stacking_adapter and emotion_adapter_name and task_name:  # if use emotion_stack is true and we have two adapters
            print(' ----- using Stack -----')
            model.active_adapters = Stack(emotion_adapter_name, task_name)
            #model.set_active_adapters([emotion_adapter_name, task_name])
        else:  # Otherwise just set them to active
            model.set_active_adapters(active_adapters_list)

        if model_args.train_all_gates_adapters:  # all gates of the adapters will be trainable, by default only the trainable adapters will have trainable gates
            names = [n for n, p in model.named_parameters()]
            paramsis = [param for param in model.parameters()]
            for n, p in zip(names, paramsis):
                if 'adapters' in n and 'gate' in n:
                    p.requires_grad = True
        

    else:
        except_para_l = []
        if config.tune_bias:
            except_para_l.append('bias')
        if config.add_lora:
            except_para_l.append('lora')
        if any([config.add_enc_prefix, config.add_dec_prefix, config.add_cross_prefix]):
            except_para_l.append('prefix')
        if len(except_para_l) > 0:
            freeze_params(model, except_para_l=except_para_l)
        elif model_args.drop_first_layers > 0:
            freeze_params_by_layers(model, num_layers, num_frozen_layers=model_args.drop_first_layers)

        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable_params: {trainable_params}, total_params: {total_params}, percentage:  {(trainable_params/total_params)*100}")

    log_wandb({'trainable_params':trainable_params, 'trainable_params_percentage':trainable_params/total_params*100}, use_wandb)
    if True:
            names = [n for n, p in model.named_parameters()]
            paramsis = [param for param in model.parameters()]
            for n, p in zip(names, paramsis):
                print(f"{n}: {p.requires_grad}")
            print(model)

    # TODO: I dont think we need this here
    #if data_args.task_name is not None:
    #    sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    #else:
    #    # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
    #    non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
    #    if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
    #        sentence1_key, sentence2_key = "sentence1", "sentence2"
    #    else:
    #        if len(non_label_column_names) >= 2:
    #            sentence1_key, sentence2_key = non_label_column_names[:2]
    #        else:
    #            sentence1_key, sentence2_key = non_label_column_names[0], None



    # Some models have set the order of the labels to use, so let's make sure we do use it.
    #label_to_id = None
    #if (
    #        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
    #        and data_args.task_name is not None
    #        and not is_regression
    #):
        # Some have all caps in their config, some don't.
    #    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
    #    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
    #        label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
    #    else:
    #        logger.warn(
    #            "Your model seems to have been trained with labels, but they don't match the dataset: ",
    #            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
    #            "\nIgnoring the model labels as a result.",
    #        )
    #elif data_args.task_name is None and not is_regression:
    #    label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # training dataset
    train_dataset_all = train_dataset.shuffle(seed=data_args.data_seed)  # this is only needed if max_train_samples is used
    if training_args.do_train:
        if train_dataset is None:
            raise ValueError("--do_train requires a train dataset dataset")
        if data_args.max_train_samples is not None:
            logger.warning(f'shuffling training set w. seed {data_args.data_seed}!')
            train_dataset = train_dataset_all.select(range(data_args.max_train_samples))

    # evaluation datset
    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("--do_eval requires a dev / evaluation dataset")
        eval_dataset = eval_dataset
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if train_dataset is None:
            raise ValueError("--do_predict requires a test dataset")
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    if data_args.train_as_val:
        test_dataset = eval_dataset
        eval_dataset = train_dataset_all.select(range(data_args.max_train_samples, data_args.max_train_samples + 1000))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # use own metric function
    # create pearson correlation as metric function
    def pearson_corr(p: EvalPrediction):
        """Correlate prediciton and true value using pearson r

        Args:
            y_pred (array): The predicted labels
            y_true (array): The true labels

        Returns:
            r, p (float, float): pearson r, p-value
        """
        r, p = pearsonr(p.label_ids.reshape((-1,)), p.predictions.reshape((-1,)))
        return r

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item(), "pearsonr": pearson_corr(p)}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        do_save_full_model=True,  # otherwise, P in AP may not be saved
        do_save_adapters=adapter_args.train_adapter
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        #print('\n -------------------------- trainer state', trainer.state.log_history)
        print(metrics)
        # print(model)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        log_wandb(metrics, use_wandb)  # Added by Myra Z.: log wandb is use_wandb == True


    #log_plot_gradients(model, tensorboard_writer, use_wandb)
    log_plot_gates(model, tensorboard_writer, use_wandb, output_dir=training_args.output_dir)
    log_plot_gates_per_layer(model, tensorboard_writer, use_wandb, output_dir=training_args.output_dir)
    log_plot_gates_per_epoch(model, tensorboard_writer, use_wandb, output_dir=training_args.output_dir)


    # set epoch of trainer state control to None so we know that training is over
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        #if data_args.task_name == "mnli":
        #    tasks.append("mnli-mm")
        #    eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            # Added by Myra Z.
            #if tensorboard_writer is not None:
            #    for met in metrics.keys():
            #        tensorboard_writer.add_scalar(str(met) + '/eval', metrics[met])

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            log_wandb(metrics, use_wandb)  # Added by Myra Z.: log wandb is use_wandb == True
            
            #predictions = trainer.predict(test_dataset=eval_dataset).predictions
            output, eval_gates_df = trainer.predict(test_dataset=eval_dataset, return_gates=True)
            predictions = output.predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            print('type eval_gates_df', type(eval_gates_df))
            print('eval_gates_df', eval_gates_df)
            try:
                print(len(eval_gates_df))
            except:
                pass
            true_score = np.reshape(eval_dataset['label'],(-1,))
            print(true_score.shape)
            try:
                essay_ids = np.reshape(eval_dataset['message_id'],(-1,))
                print(essay_ids.shape)

            except:
                pass
            log_plot_predictions(true_score, predictions, tensorboard_writer, use_wandb)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        # not evaluating test_mismatched
        # if data_args.task_name == "mnli":
        #     tasks.append("mnli-mm")
        #     test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # only do_predict if train_as_val
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # test_dataset.remove_columns_("label")
            metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix='test')
            trainer.log_metrics("test", metrics)
            trainer.save_metrics("test", metrics)
            log_wandb(metrics, use_wandb)  # Added by Myra Z.: log wandb is use_wandb == True

            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            
            # Added by Myra Z.
            true_score = np.reshape(test_dataset['label'],(-1,))
            log_plot_predictions(true_score, predictions, tensorboard_writer, use_wandb, output_dir=training_args.output_dir, split='test')

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    
    # Added by Myra Z.
    if len(model.bert.gates) > 0:
        model.bert.gates.to_csv(training_args.output_dir + '/gates.csv')
    


# Added by Myra Z.
def log_plot_predictions(y_true, y_hat, tensorboard_writer, use_wandb=False, output_dir='', split=''):
    plt.scatter(y_true, y_hat)
    plt.xlabel('true score')
    plt.ylabel('predictions')
    plt.title(f'Scatterplot of the true labels and the predictions of {split}.')
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Scatter_Predictions', plt.gcf())
    if use_wandb:
        title = split + '_predictions'
        wandb.log({title: wandb.Image(plt)})
    if os.path.exists(output_dir) and output_dir != '':
        plt.savefig(output_dir + '/' + split + '_predictions.pdf', bbox_inches='tight')
    plt.close()

def log_wandb(metrics, use_wandb):
    if use_wandb:  # only log if True, otherwise package might not be available
        wandb.log(metrics)  # Added by Myra Z.

def log_plot_gradients(model, tensorboard_writer, use_wandb=False, output_dir=''):
    # plot gating of the gradients of the last state of the model
    # investigate gating
    grad_by_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'gat' in name and not 'bias' in name:
            gate_grad = param.data.detach().squeeze().cpu().numpy()
            # get the number of the layer using regex
            layer = re.search('layer\.[0-9]{1,}', name)
            if layer is None: continue  # just go to next step and do nothing
            layer_num = re.search('[0-9]{1,}', layer.group(0)).group(0)
            # make dictionary for each layer with the different elements as touples:
            # (dict(str:(str, np.array)), e.g. {'1': ('layer1.lora', np.array([1,2]))}))
            if layer_num not in grad_by_layer:
                grad_by_layer[layer_num] = [(name, gate_grad)]
            else:
                grad_by_layer[layer_num].append((name, gate_grad))

    # create figure from the gating layers
    sub_fig_y = len(grad_by_layer.keys())
    sub_fig_x = max([len(grad_by_layer[key]) for key in grad_by_layer.keys()])

    fig, axs = plt.subplots(sub_fig_y, sub_fig_x, sharey=True, sharex=True, figsize=(sub_fig_x*5, sub_fig_y*5))

    for i, (key, val) in enumerate(sorted(grad_by_layer.items(), key=lambda x: int(x[0]))):
        key_num = int(key)
        for j, (name, grads) in enumerate(val):
            axs[i, j].hist(grads, orientation="horizontal")
            axs[i,j].set_title(name)
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Gating gradients per layer', plt.gcf())
    if use_wandb:
        wandb.log({'Gating gradients per layer': wandb.Image(plt)})
    if os.path.exists(output_dir) and output_dir != '':
        plt.savefig(output_dir + '/' + 'gradient_gating_per_layer.pdf', bbox_inches='tight')

    plt.close()

    fig, axs = plt.subplots(sub_fig_y, 1, sharey=True, sharex=False, figsize=(sub_fig_x*5/2, sub_fig_y*5))

    for i, (key, val) in enumerate(sorted(grad_by_layer.items(), key=lambda x: int(x[0]))):
        key_num = int(key)
        gradients = [grads for (n, grads) in val]
        names = [n for (n, grads) in val]
        axs[i].boxplot(gradients, labels=names)
    
    if tensorboard_writer is not None:
        tensorboard_writer.add_figure('Gating gradients per layer', plt.gcf())
    if use_wandb:
        wandb.log({'Gating gradients per layer: Boxplot': wandb.Image(plt)})
    plt.close()


def log_plot_gates(model, tensorboard_writer, use_wandb=False, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))
    # get eval and train of last epoch while in train
    last_train = gates[(gates['split'] == 'train_evaluation') & (gates['epoch'] == max(gates['epoch'])) & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == False)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    count_data_available = sum([1 for key in gate_per_set.keys() if show_plot_crit(key)])

    for layer in encoder_layers:
        fig, axs = plt.subplots(count_data_available, sharey=True, sharex=False, constrained_layout=True)
        idx = 0
        for key in gate_per_set.keys():
            columns = ['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters']
            if show_plot_crit(key):
                #print(gate_per_set[key]) 
                dataset = gate_per_set[key]
                dataset = dataset[dataset['encoder_layer'] == layer].reset_index()
                dataset.dropna(axis=1, inplace=True)
               
                if count_data_available == 1:
                    for i, col in enumerate(columns):
                        if col not in dataset.columns:
                            continue
                        axs.plot(dataset[col].to_numpy(), label=col[5:], c=COLORS[i])
                    axs.set_ylabel('gating value')
                    axs.set_title(f'{key} data set')
                else:
                    for i, col in enumerate(columns):
                        if col not in dataset.columns:
                            continue
                        axs[idx].plot(dataset[col].to_numpy(), label=col[5:], c=COLORS[i])
                    #axs[idx].plot(dataset[['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters']], label=['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters'])
                    axs[idx].set_ylabel('gating value')
                    axs[idx].set_title(f'{key} data set')
                idx += 1

        title = f'{key}/gating/layer{int(layer) + 1}'
        fig.suptitle(title)
        plt.legend()
        plt.xlabel('data sample')
        plt.ylim(0,1)
        if tensorboard_writer is not None:
            tensorboard_writer.add_figure(title, plt.gcf())
        if use_wandb:
            wandb.log({title: wandb.Image(plt)})
        if os.path.exists(output_dir) and output_dir != '':
            plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
        plt.close()


def log_plot_gates_per_layer(model, tensorboard_writer, use_wandb, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))
    # get eval and train of last epoch while in train
    last_train = gates[(gates['split'] == 'train') & (gates['epoch'] == max(gates['epoch'])) & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == False)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    count_data_available = sum([1 for key in gate_per_set.keys() if show_plot_crit(key)])

    idx = 0
    for key in gate_per_set.keys():
        if show_plot_crit(key):
            fig, axs = plt.subplots()
            dataset = gate_per_set[key]
            #dataset = dataset[dataset['encoder_layer'] == layer].reset_index()

            gating_cols = [col for col in dataset.columns if 'gate' in col]
            grouped_mean = dataset.groupby(['encoder_layer']).agg({col: 'mean' for col in gating_cols})
            grouped_std = dataset.groupby(['encoder_layer']).agg({col: 'std' for col in gating_cols})

            bar_width = 2
            width = bar_width * (len(gating_cols)) + bar_width
            x = grouped_mean.index.to_numpy() * width
            this_colors = COLORS
            if len(this_colors) < len(gating_cols):  # if not enough colors in the list
                this_colors = [COLORS[i] if i < len(COLORS) else '#000000' for i in range(len(gating_cols))]

            fig, axs = plt.subplots()
            fig.set_figheight(len(grouped_mean))
            for idx, col in enumerate(gating_cols):
                y_pos = x + idx * bar_width
                color_i = this_colors[idx]
                label_i = col[5:].replace('-', ' ').replace('_', ' ')
                axs.barh(y=y_pos, xerr=grouped_std[col], width=grouped_mean[col], height=bar_width, label=label_i, color=color_i)#, 'gate_lora_value', 'gate_lora_query', 'gate_adapters']], label=['gate_prefix', 'gate_lora_value', 'gate_lora_query', 'gate_adapters'])
            
            axs.set_ylabel('Encoder Layer')
            axs.set_yticklabels(grouped_mean.index.to_numpy())
            axs.set_yticks(x + ((len(gating_cols)-1) * bar_width)/2)
            axs.set_title(f'{key} data set')
            axs.set_ylim(x[0]-bar_width/2, x[-1] + ((len(gating_cols)-1) * bar_width) + bar_width/2)
            axs.legend()#loc='center left', bbox_to_anchor=(1, 0.5))


            title = f'Mean Gating Values for all Encoder Layers'
            fig.suptitle(title)
            title = f'{key}/gating_layers'
            plt.xlabel('Mean gating value')
            if tensorboard_writer is not None:
                tensorboard_writer.add_figure(title, plt.gcf())
            if use_wandb:
                wandb.log({title: wandb.Image(plt)})
            if os.path.exists(output_dir) and output_dir != '':
                plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
            plt.close()

def log_plot_gates_per_epoch(model, tensorboard_writer=None, use_wandb=False, output_dir=''):
    gates = model.bert.gates
    if gates.empty:
        return
    encoder_layers = sorted(set(gates['encoder_layer']))

    last_train = gates[(gates['split'] == 'train') & (gates['is_in_train'] == True)].reset_index()
    after_train_eval = gates[(gates['split'] == 'eval') & (gates['is_in_train'] == True)].reset_index()
    after_train_test = gates[(gates['split'] == 'test') & (gates['is_in_train'] == False)].reset_index()

    show_plot_crit = lambda key: len(gate_per_set[key]) > 0 if key in gate_per_set.keys() else False # criterion to not show the plot for the data set, here: if dataset not used / df is empty
    gate_per_set = {'train':last_train, 'eval':after_train_eval, 'test':after_train_test}
    
    idx = 0
    for key in gate_per_set.keys():
        if show_plot_crit(key):
            dataset = gate_per_set[key]
            dataset.dropna(axis=1, inplace=True)
            #available_columns = [c for c in columns if c in dataset.columns]
            #if len(available_columns) < 2:  # No column available -> no plot
            #    continue
            gating_cols = [col for col in dataset.columns if 'gate' in col]
            grouped_mean = dataset.groupby(['encoder_layer', 'epoch']).agg({col: 'mean' for col in gating_cols})
            grouped_std = dataset.groupby(['encoder_layer', 'epoch']).agg({col: 'std' for col in gating_cols})
            #grouped_mean = dataset.groupby(['encoder_layer', 'epoch']).agg({c: 'mean' for c in available_columns})
            #grouped_std = dataset.groupby(['encoder_layer', 'epoch']).agg({c: 'std' for c in available_columns})
            for layer in encoder_layers:
                layer_mean = grouped_mean.loc[layer]
                layer_std = grouped_std.loc[layer]
                fig, axs = plt.subplots()
                bar_width = 2
                width = bar_width*4 + bar_width*1.5
                
                for i, col in enumerate(gating_cols):
                    if col not in gating_cols:
                        continue

                    label_i = col[5:].replace('-', ' ').replace('_', ' ')
                    x = np.array(range(len(layer_mean[col].to_numpy())))
                    axs.plot(x, layer_mean[col].to_numpy(), c=COLORS[i], label=label_i)
                    axs.fill_between(x, layer_mean[col].to_numpy() + layer_std[col].to_numpy(), layer_mean[col].to_numpy() - layer_std[col].to_numpy(), color=COLORS[i], alpha=0.5)
                if len(x) <= 1:
                    plt.close()
                    continue
                axs.set_xlabel('epochs')
                axs.set_ylabel('gating value')
                axs.set_ylim(0,1)
                axs.set_xticks(x)
                axs.set_xticklabels([str(item + 1) for item in x])
                axs.set_title(f'Layer {layer +1}: Mean of the Gating Values with Std \n {key} data')
                axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                idx += 1
                fig.tight_layout()
                
                plt.show()
                title = f'{key}/gating_plot/layer{layer + 1}'
                if tensorboard_writer is not None:
                    tensorboard_writer.add_figure(title, plt.gcf())
                if use_wandb:
                    wandb.log({title: wandb.Image(plt)})
                if os.path.exists(output_dir) and output_dir != '':
                    plt.savefig(output_dir + '/' + title.replace('/', '_') + '.pdf', bbox_inches='tight')
                plt.close()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    # --- run on GPU if available ---
    if torch.cuda.is_available():       
        #device = torch.device("cuda")
        choose_gpu(min_gpu_memory=5000, retry=False)
        print("\n------------------ Using GPU. ------------------\n")
    else:
        print("\n---------- No GPU available, using the CPU instead. ----------\n")
        #device = torch.device("cpu")


    main()
