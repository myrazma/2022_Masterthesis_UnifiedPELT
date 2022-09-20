# Usage of UniPELT repo (CLONE) for my Master's thesis

This is a clone of of the [UniPELT repository](https://github.com/morningmoni/UniPELT), see [Reference](#Reference) for the citation of the authors and original paper of UniPELT. Also refer to other repositories used in the UniPELT repo: [Prefix-tuning](https://github.com/XiangLi1999/PrefixTuning), [LoRA](https://github.com/microsoft/LoRA)

In this directory, we use the UniPELT framework and adapt it to the [task used in my Master's thesis](#The-task-Empathy-and-distress-prediction). The UniPELT framework is a unified version with Parameter-Efficient Language Model Tuning (PELT) techniques, including LoRA (Hu et al., 2021), BitFit (bias-term fine tuning) (Zaken et al., 2021), prefix tuning (Li & Liang, 2021) and adapters (Houlsby et a., 2019). These methods are combined using a gating function to activate (up-weight) the best performing method for the task setting and data sample.

For the experiments, analysis and further research, please visit the repository of my [Master's thesis](https://github.com/myrazma/2022_Masterthesis_Code)

## The task: Empathy and distress prediction
The task is a supervised regression task to predict empathy and distress ratings from texts (Buechel et al., 2018). Participants were asked to read 5 news articles. After each article, they reported their empathy and distress level on a 14-item questionnaire and wrote a reaction essay. This essay is the input for our NLP model. The labels are the average questionnaire.

## Our additions to this repository

### Changes in UniPELT:
[modeling_bert.py](transformers/models/bert/modeling_bert.py): We added storing and flushing the gate variables for more structured access to the parameters. 

In addition, we updated the model to return the gates at inference. One can access the gates via trainer.predict:
```
output, gates = trainer.predict(test_dataset=eval_dataset, return_gates=True)
```

### Changes or addition for the empathy and distress prediction directly:
* [run_emp.py](run_emp.py): copy from run_glue.py and adapted to the empathy / distress prediction
* [utils.py](utils.py): add some more functions to the already existing utils.py script
* [preprocessing.py](preprocessig.py): add a preprocessing script for useful preprocessing methods for the empathy and distress dataset
* [run_unipelt_emp.sh](run_unipelt_emp.sh): add this file to run the run_emp.py script with all necessary arguments
* [run_unipelt_emp_testing.sh](run_unipelt_emp_testing.sh): Add this file to run the run_emp.py script as a test run with less epochs and less training data
* [runs/](runs/): The runs for the tensorboard

For the other scripts, please refer to the original authors.

## Run
As a Dockerfile was not provided by the original authors, I created one with all the necessary requirements.
To build the Dockerfile run
```
docker build -t <docker-name> .
```

and run in bash (using gpus here and the container will be removed after exit)
```
docker run --gpus all --rm -it -v "$PWD":/src <docker-name> bash
```

Run UniPELT for the empathy task by running

```
bash run_unipelt_emp.sh
```

If you want to make any changes to the training, model and data setting, you can also change the parameters in that file. 

### Run settings for the UniPelt configurations
The different configurations for the models are set in [run_unipelt_emp.sh](run_unipelt_emp.sh) and can be changed by changing *pelt_method* to one of the pre-defiend cases (full, unipelt, prefix, lora, bitfit, adapter):

#### Full parameter fine-tuning (full):
Using 100 % of the parameters (for bert-base).
```
    learning_rate=2e-5
    tensorboard_output_dir=runs/pelt_full_fine_tuning
    add_enc_prefix=False
    train_adapter=False
    add_lora=False
    tune_bias=False
```

#### UniPELT (Lora, Adapters, BitFit, Prefix)
```
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_unified
    add_enc_prefix=True
    train_adapter=True
    add_lora=True
    tune_bias=True
```

#### BitFit: train bias only (bitfit):
```
    learning_rate=1e-3
    tensorboard_output_dir=runs/pelt_bitfit
    add_enc_prefix=False
    train_adapter=False
    add_lora=False
    tune_bias=True
```

#### LoRA (lora)
```
    learning_rate=5e-4
    tensorboard_output_dir=runs/pelt_lora
    add_enc_prefix=False
    train_adapter=False
    add_lora=True
    tune_bias=False
```

#### Prefix-Tuning (prefix)
```
    learning_rate=2e-4
    tensorboard_output_dir=runs/pelt_prefix
    add_enc_prefix=True
    train_adapter=False
    add_lora=False
    tune_bias=False
```

#### Adapter (adapter)
```
    learning_rate=1e-4
    tensorboard_output_dir=runs/pelt_adapters
    add_enc_prefix=False
    train_adapter=True
    add_lora=False
    tune_bias=False
```


# Bibliography
Buechel, S., Buffone, A., Slaff, B., Ungar, L., & Sedoc, J. (2018). Modeling empathy and distress in reaction to news stories. arXiv preprint arXiv:1808.10399. https:
//arxiv.org/abs/1808.10399

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., & Gelly, S. (2019). Parameter-efficient transfer learning for nlp. International Conference on Machine Learning, 2790–2799. http://proceedings. mlr.press/v97/houlsby19a/houlsby19a.pdf

Hu, E. J., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. (2021). Lora: Low-rank adaptation of large language models. International Conference on Learning Representations. https://arxiv.org/abs/2106.09685

Li, X. L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190. https://arxiv.org/abs/2101.00190

Zaken, E. B., Goldberg, Y., & Ravfogel, S. (2022). Bitfit: Simple parameter-efficient fine- tuning for transformer-based masked language-models. Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 1–9. https://aclanthology.org/2022.acl-short.1/




## README.md from the original authors of UniPELT
This repo provides the code for paper ["UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning"](https://arxiv.org/abs/2110.07577), ACL 2022.

We support multiple Parameter-Efficient Language model Tuning (PELT) methods, including Prefix-tuning, Adapter, LoRA, BitFit, and any combination of them on BERT.



### How to run
Use `run_glue.py` as the entry file.

To use Prefix-tuning, set `--add_enc_prefix True`

To use Adapter, set `--train_adapter`

To use LoRA, set `--add_lora True`

To use BitFit, set `--tune_bias True`

The codebase is based on [transformers (adapter-transformers)](https://github.com/Adapter-Hub/adapter-transformers/). See [here](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L86) for more details of the training arguments.
Please also refer to the following repos: [Prefix-tuning](https://github.com/XiangLi1999/PrefixTuning), [LoRA](https://github.com/microsoft/LoRA).

### Reference
Reference for UniPELT implementation.
```
@inproceedings{mao-etal-2022-unipelt,
  title={UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning},
  author={Mao, Yuning and Mathias, Lambert and Hou, Rui and Almahairi, Amjad and Ma, Hao and Han, Jiawei and Yih, Wen-tau and Khabsa, Madian},
  journal={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

### License
The majority of UniPELT is licensed under CC-BY-NC, however portions of the project are available under separate license terms: transformers (adapter-transformers) is licensed under the Apache 2.0 license and LoRA is licensed under the MIT License.

