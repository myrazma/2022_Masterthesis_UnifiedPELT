# Parmater setup from Mao2021 - Unipelt: A unified framework for parameter-efficient language model tuning
# https://arxiv.org/abs/2110.07577

# Run unipelt experiment for all methods

# setup wandb
wandb_entity="masterthesis-zmarsly"
wandb_project="Results"
# or to not use wandb use:
#wandb_entity="None"

# Differences to parameters from UniPELT:
# max_seq_length = 256 instead of 128

# choose a method here and use the settings as stated below
do_predict=False
#task_names=( distress empathy )
task_names=( empathy )
#methods=( full unipelt unipelt_apl unipelt_ap unipelt_al adapter lora prefix bitfit )
methods=( unipelt unipelt_apl unipelt_ap unipelt_al adapter lora prefix bitfit )
range_runs=$((${#methods[@]}*${#task_names[@]}))
i=0

for task_name in "${task_names[@]}"
do
    for pelt_method in "${methods[@]}"
    do 
        i=$(($i+1))
        echo "--------------- Run $i of $range_runs ---------------"
        echo "----------- $task_name using $pelt_method -----------"

        # Full fine tuning
        if [ $pelt_method == "full" ]; then
            echo "Using Full fine tuning"
            learning_rate=2e-5
            tensorboard_output_dir=runs/pelt_full_fine_tuning_bert
            output_dir=output/pelt_full_fine_tuning_bert
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
            output_dir=output/pelt_unified_aplb_bert
            add_enc_prefix=True
            train_adapter=True
            add_lora=True
            tune_bias=True
        fi

        # UniPELT APL
        if [ $pelt_method == "unipelt_apl" ]; then
            echo "Using Unipelt APL (adapter, prefix-tuning, lora; exclude: BitFit)"
            learning_rate=1e-4
            tensorboard_output_dir=runs/pelt_unified_apl_bert
            output_dir=output/pelt_unified_apl_bert
            add_enc_prefix=True
            train_adapter=True
            add_lora=True
            tune_bias=False
        fi

        # UniPELT AP
        if [ $pelt_method == "unipelt_ap" ]; then
            echo "Using Unipelt APL (adapter, prefix-tuning; exclude: LoRA, BitFit)"
            learning_rate=5e-4
            tensorboard_output_dir=runs/pelt_unified_ap_bert
            output_dir=output/pelt_unified_ap_bert
            add_enc_prefix=True
            train_adapter=True
            add_lora=False
            tune_bias=False
        fi        
        
        # UniPELT AL
        if [ $pelt_method == "unipelt_al" ]; then
            echo "Using Unipelt APL (adapter, LoRA; exclude: prefix-tuning, BitFit)"
            learning_rate=5e-4
            tensorboard_output_dir=runs/pelt_unified_al_bert
            output_dir=output/pelt_unified_al_bert
            add_enc_prefix=False
            train_adapter=True
            add_lora=True
            tune_bias=False
        fi

        # LoRA
        if [ $pelt_method == "lora" ]; then
            echo "Using LoRA"
            learning_rate=5e-4
            tensorboard_output_dir=runs/pelt_lora_bert
            output_dir=output/pelt_lora_bert
            add_enc_prefix=False
            train_adapter=False
            add_lora=True
            tune_bias=False
        fi

        # BitFit
        if [ $pelt_method == "bitfit" ]; then
            echo "Using BitFit"
            learning_rate=1e-3
            tensorboard_output_dir=runs/pelt_bitfit_bert
            output_dir=output/pelt_bitfit_bert
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
            output_dir=output/pelt_prefix
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
            output_dir=output/pelt_adapters_bert
            add_enc_prefix=False
            train_adapter=True
            add_lora=False
            tune_bias=False
        fi

        # add task name as subfolder
        output_dir="${output_dir}/${task_name}"

        # call the python file with stated parameters
        python run_emp.py \
            --task_name ${task_name} \
            --data_dir data/ \
            --output_dir ${output_dir} \
            --overwrite_output_dir \
            --model_name_or_path bert-base-uncased \
            --do_predict ${do_predict} \
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
            --wandb_project ${wandb_project} \
            --use_tensorboard False \
            --tensorboard_output_dir ${tensorboard_output_dir} \
            --add_enc_prefix ${add_enc_prefix} \
            --train_adapter ${train_adapter} \
            --add_lora ${add_lora} \
            --tune_bias ${tune_bias} \
            --learning_rate ${learning_rate} \

    done
done