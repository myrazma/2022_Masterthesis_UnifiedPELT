# Outsource argument dataclasses

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "empathy": ("sentence", None),  # edited by Myra Z.
    "distress": ("sentence", None)  # edited by Myra Z.
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    # edited by Myra Z.
    data_dir: str = field(
        default='data', metadata={"help": "A directory containing the data."}
    )
    # edited by Myra Z.
    task_name: Optional[str] = field(
        default='distress',
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    # edited by Myra Z.
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )
    # added by Myra Z.
    wandb_entity: Optional[str] = field(
        default=None, metadata={"help": "The entity name of the wandb user. Leave empty if you do not wish to use wandb"}
    )
    # added by Myra Z.
    wandb_project: str = field(
        default="Results", metadata={"help": "The Project of the wandb."}
    )
    # added by Myra Z.
    use_tensorboard: Optional[bool] = field(
        default=False, metadata={"help": "If True, use a writer for tensorboard"}
    )
    # added by Myra Z.
    tensorboard_output_dir: str = field(
        default="runs/", metadata={"help": "Path to the sub directory of the writer. Saves in runs/ + output_dir"}
    )
    # edited by Myra Z.
    train_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/messages_train_ready_for_WS.tsv', metadata={"help": "A csv or a json file containing the training data."}
    )
    # edited by Myra Z.
    validation_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/messages_dev_features_ready_for_WS_2022.tsv', metadata={"help": "A csv or a json file containing the validation data."}
    )
    # edited by Myra Z.
    validation_labels_file: Optional[str] = field(
        default=data_dir.default + '/buechel_empathy/goldstandard_dev_2022.tsv', metadata={"help": "A csv or a json file containing the validation data."}
    )
    # edited by Myra Z.
    test_file: Optional[str] = field(default=data_dir.default + '/buechel_empathy/messages_test_features_ready_for_WS_2022.tsv', metadata={"help": "A csv or a json file containing the test data."})
    data_seed: Optional[int] = field(
        default=42,
        metadata={
            "help": "seed for selecting subset of the dataset if not using all."
        },
    )
    train_as_val: bool = field(
        default=False,
        metadata={"help": "if True, sample 1k from train as val"},
    )
    early_stopping_patience: Optional[int] = field(
        default=10,
    )    

    def __post_init__(self):
        # overwritten by Myra Z.
        if not os.path.exists(self.data_dir):  # Addition from Myra Zmarsly
            raise ValueError(f"The data directory: {self.data_dir} does not exists.")
        elif not os.listdir(self.data_dir):
            raise ValueError(f"The data directory {self.data_dir} is empty.")
        elif (not os.path.exists(self.train_file)) or (not os.path.exists(self.validation_file))or (not os.path.exists(self.validation_labels_file)):
            raise ValueError(f"The buechel_empathy data does not exist {self.data_dir} or is not stored / named corretly. The data should be in dir /buechel_empathy/")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='bert-base-uncased', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    # prefix-tuning parameters
    add_enc_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_dec_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    add_cross_prefix: bool = field(
        default=False,
        metadata={"help": "Whether use prefix tuning"},
    )
    prefix_len: Optional[int] = field(
        default=10,
        metadata={"help": "length of prefix tokens"},
    )
    mid_dim: Optional[int] = field(
        default=512,
        metadata={"help": "dim of middle layer"},
    )
    # bitfit parameters
    tune_bias: bool = field(
        default=False,
        metadata={"help": "Whether tune bias terms"},
    )
    # LoRA parameters
    add_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use lora for linear layers"},
    )
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": "rank of lora"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "scaling = alpha / r"},
    )

    drop_first_layers: Optional[int] = field(
        default=0,
        metadata={
            "help": "drop first k layers, work for both prefix and adapter, freeze transformer layers if fine-tuning"},
    )
    drop_first_adapter_layers: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k adapter layers"},
    )
    drop_first_prefix_layers_enc: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_dec: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    drop_first_prefix_layers_cross: Optional[int] = field(
        default=0,
        metadata={"help": "drop first k prefix layers"},
    )
    add_adapter_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the adapter"},
    )
    add_prefix_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the prefix"},
    )
    add_lora_gate: bool = field(
        default=True,
        metadata={"help": "add a gate to the lora"},
    )
    add_central_gate: bool = field(
        default=False,
        metadata={"help": "add a shared gate"},
    )
    stacking_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "The source of an adapter to stack right before the task adapter"},
    )
    stacking_adapter: Optional[str] = field(
        default=None,
        metadata={"help": "The source of an adapter to stack right before the task adapter"},
    )
    use_stacking_adapter: bool = field(
        default=False,
        metadata={"help": "The source of an adapter to stack right before the task adapter"},
    )
    train_all_gates_adapters: bool = field(
        default=True,
        metadata={"help": "Train the gate for all adapters, even if they are not set to active"},
    )
