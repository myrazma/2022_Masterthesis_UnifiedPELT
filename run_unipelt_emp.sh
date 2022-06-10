# Parmater setup from Mao2021 - Unipelt: A unified framework for parameter-efficient language model tuning
# https://arxiv.org/abs/2110.07577

# setup wandb
wandb_entity="masterthesis-zmarsly"
# or to not use wandb use:
#wandb_entity="None"

# Differences to parameters from UniPELT:
# max_seq_length = 256 instead of 128

# choose a method here and use the settings as stated below
#pelt_method="full"
#pelt_method="unipelt"
pelt_method="unipelt_apl"
#pelt_method="unipelt_ap"
#pelt_method="adapter"
#pelt_method="lora"
#pelt_method="prefix"
#pelt_method="bitfit"

# Full fine tuning
if [ $pelt_method == "full" ]; then
    echo "Using Full fine tuning"
    learning_rate=2e-5
    tensorboard_output_dir=runs/pelt_full_fine_tuning_bert
    add_enc_prefix=False
    train_adapter=False
    add_lora=False
    tune_bias=False
fi

# UniPELT
if [ $pelt_method == "unipelt" ]; then
    echo "Using Unipelt (Prefix, adapter, lora, bitfit)"
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_unified_aplb_bert
    add_enc_prefix=True
    train_adapter=True
    add_lora=True
    tune_bias=True
fi

# UniPELT APL
if [ $pelt_method == "unipelt_apl" ]; then
    echo "Using Unipelt APL (adapter, prefix-tuning, lora; exclude: BitFit)"
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_unified_apl_bert
    add_enc_prefix=True
    train_adapter=True
    add_lora=True
    tune_bias=False
fi

# UniPELT APL
if [ $pelt_method == "unipelt_ap" ]; then
    echo "Using Unipelt APL (adapter, prefix-tuning; exclude: LoRA, BitFit)"
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_unified_ap_bert
    add_enc_prefix=True
    train_adapter=True
    add_lora=False
    tune_bias=False
fi

# LoRA
if [ $pelt_method == "lora" ]; then
    echo "Using LoRA"
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_lora_bert
    add_enc_prefix=False
    train_adapter=False
    add_lora=True
    tune_bias=False
fi

# BitFit
if [ $pelt_method == "bitfit" ]; then
    echo "Using BitFit"
    learning_rate=1e-4
    tensorboard_output_dir=runs/pelt_bitfit_bert
    add_enc_prefix=False
    train_adapter=False
    add_lora=False
    tune_bias=True
fi

# Prefix-Tuning
if [ $pelt_method == "prefix" ]; then
    echo "Using Prefix-tuning"
    learning_rate=2e-4
    tensorboard_output_dir=runs/pelt_prefix
    add_enc_prefix=True
    train_adapter=False
    add_lora=False
    tune_bias=False
fi

# Adapters
if [ $pelt_method == "adapter" ]; then
    echo "Using adapter"
    learning_rate=1e-4
    tensorboard_output_dir=runs/pelt_adapters_bert
    add_enc_prefix=False
    train_adapter=True
    add_lora=False
    tune_bias=False
fi

# call the python file with stated parameters
python run_emp.py \
    --data_dir data/ \
    --output_dir output/unipelt_output  \
    --overwrite_output_dir \
    --model_name_or_path bert-base-uncased \
    --do_predict False \
    --do_eval True \
    --do_train True \
    --num_train_epochs 15 \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size 16 \
    --early_stopping_patience 5 \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy no \
    --wandb_entity ${wandb_entity} \
    --use_tensorboard False\
    --tensorboard_output_dir ${tensorboard_output_dir} \
    --add_enc_prefix ${add_enc_prefix} \
    --train_adapter ${train_adapter} \
    --add_lora ${add_lora} \
    --tune_bias ${tune_bias} \
    --learning_rate ${learning_rate} \