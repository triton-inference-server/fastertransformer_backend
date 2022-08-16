<!--
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
-->

# FasterTransformer GPT-NeoX Triton Backend

Associated GPT-NeoX documentation for the FasterTransformer side can be found at [gptneox_guide.md](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gptneox_guide.md).

## Table Of Contents

- [FasterTransformer GPT-NeoX Triton Backend](#fastertransformer-gpt-neox-triton-backend)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup Environment](#setup-environment)
    - [How to set the model configuration](#how-to-set-the-model-configuration)
    - [Decoupled mode](#decoupled-mode)
    - [Prepare Triton GPT-NeoX model store](#prepare-triton-gpt-neox-model-store)
  - [Run Serving on Single Node](#run-serving-on-single-node)
    - [Run serving directly](#run-serving-directly)
      - [Run GPT-NeoX end-to-end serving by Triton ensemble](#run-gpt-neox-end-to-end-serving-by-triton-ensemble)
      - [Evaluate the accuracy of GPT-NeoX model on LAMBADA.](#evaluate-the-accuracy-of-gpt-neox-model-on-lambada)

## Introduction

This document describes how to serve the `GPT-NeoX` model by FasterTransformer Triton backend. This backend is only an interface to call FasterTransformer in Triton. All implementation are in [FasterTransformer repo](https://github.com/NVIDIA/FasterTransformer).

## Setup Environment

Follow the guide in [`README.md`](../README.md) to setup the environment and prepare docker image. We assume users already built the docker image here.

### How to set the model configuration

Generally, we need two configuration files to server the FasterTransformer models.

**Model Configuration: config.ini generated during converting the model**

  Normally, this is will be generated automatically when you converting the model checkpoint to FasterTransformer format. However, some configurations (like start_id, end_id) may need to be modified on your own.
  It is because the converter doesn't know anything about the tokenizer if the original checkpoint configurations don't contain such information.

  We provide an example in `all_models/gptneox/fastertransformer/1/config.ini`.

  - This should be placed in the same directory of model weights
  - This will be loaded by fastertransformers.
  - This mainly describes the model structure and prompt hyperparameters, start_id, end_id, and so on.

  The following table shows the details of config.ini:

  | Classification |             Name             | Tensor/Parameter Shape         | Data Type |                                                    Description                                                    |
  | :------------: | :--------------------------: | :----------------------------- | :-------: | :---------------------------------------------------------------------------------------------------------------: |
  |      gptneox   |                              |                                |           |                                                                                                               |
  |                |       `max_pos_seq_len`      |                                |    int    |                          maximum sequence length supported for position embedding table  (only needed by absolute position embedding) |
  |                |          `head_num`          |                                |    int    |                    the number of head in transformer attention block. A model hyper-parameter                     |
  |                |       `size_per_head`        |                                |    int    |                   the size of each head in transformer attention block. A model hyper-parameter                   |
  |                |         `inter_size`         |                                |    int    |                            the intermediate size of FFN layer. A model hyper-parameter                            |
  |                |         `vocab_size`         |                                |    int    |                                              the size of vocabulary.                                              |
  |                |          `start_id`          |                                |    int    |          the id for start token for un-conditional generation task. In GPT-J, it is often same to end_id          |
  |                |         `num_layer`          |                                |    int    |                             the number of transformer layer. A model hyper-parameter                              |
  |                |      `use_gptj_residual`     |                                |    int    |                                  use the gptj residual style or the normal gpt style                              |
  |                |      `rotary_embedding`      |                                |    int    |                                  rotary embedding size. A model hyper-parameter                                   |
  |weight_data_type|      `weight_data_type`      |                                |    str    |                                 the weight data type (stored in fastertransformer format), and  will be casted when loaded if necessary |
  |prompt_learning |                              |                                |           |                                                                                                                   |
  |                |    `prompt_learning_type`    |                                |    int    |          the prompt learning type: [0] no prompt [1] soft prompt [2] prefix_prompt [3] p/prompt tuning            |
  |                | `prompt_learning_start_id`   |                                |    int    |          the prompt learning virtual token start id: only used by p/prompt_tuning to check if id is a prompt or not |
  |task_i          |                              |                                |           |                            the prompt learning task: task Name id = i (0, 1, ....)                                |
  |                |      `task_name`             |                                |    str    |                              the task_name used to load specific prompt weights                                   |
  |                |      `prompt_length`         |                                |    int    |                              the prompt tokens total length                                                       |


**Fastertransformer-Triton Serving Configuration: config.pbtxt**

In GPT-NeoX triton backend, the serving configuration is controlled by `config.pbtxt`. We provide an example in `all_models/gptneox/fastertransformer/config.pbtxt`. It contains the input parameters, output parameters, and some other settings like `tensor_para_size` and `model_checkpoint_path`.

The following table shows the details of these settings:

* Settings in config.pbtxt

| Classification |             Name             | Tensor/Parameter Shape         | Data Type |                                                    Description                                                    |
| :------------: | :--------------------------: | :----------------------------- | :-------: | :---------------------------------------------------------------------------------------------------------------: |
|     input      |                              |                                |           |                                                                                                                   |
|                |         `input_ids`          | [batch_size, max_input_length] |  uint32   |                                           input ids after tokenization                                            |
|                |       `input_lengths`        | [batch_size]                   |  uint32   |                                        real sequence length of each input                                         |
|                |     `request_output_len`     | [batch_size]                   |  uint32   |                                        how many tokens we want to generate                                        |
|                |       `runtime_top_k`        | [batch_size]                   |  uint32   |                                           candidate number for sampling                                           |
|                |       `runtime_top_p`        | [batch_size]                   |   float   |                                         candidate threshold for sampling                                         |
|                | `beam_search_diversity_rate` | [batch_size]                   |   float   |               diversity rate for beam search in this [paper](https://arxiv.org/pdf/1611.08562.pdf)                |
|                |        `temperature`         | [batch_size]                   |   float   |                                               temperature for logit                                               |
|                |        `len_penalty`         | [batch_size]                   |   float   |                                             length penalty for logit                                              |
|                |     `repetition_penalty`     | [batch_size]                   |   float   |                                           repetition penalty for logit                                            |
|                |        `random_seed`         | [batch_size]                   |  uint64   |                                             random seed for sampling                                              |
|                |    `is_return_log_probs`     | [batch_size]                   |   bool    |                              flag to return the log probs of generated token or not.                              |
|                |         `beam_width`         | [batch_size]                   |  uint32   |                             beam size for beam search, using sampling if setting to 1                             |
|                |       `bad_words_list`       | [batch_size, 2, word_list_len] |   int32   |  List of tokens (words) to never sample. Should be generated with `all_models/gpt/preprocessing/1/word_list.py`   |
|                |      `stop_words_list`       | [batch_size, 2, word_list_len] |   int32   | List of tokens (words) that stop sampling. Should be generated with `all_models/gpt/preprocessing/1/word_list.py` |
|     output     |                              |                                |           |                                                                                                                   |
|                |         `output_ids`         | [batch_size, beam_width, -1]   |  uint32   |                                         output ids before detokenization                                          |
|                |      `sequence_length`       | [batch_size, beam_width]       |  uint32   |                                       final sequence lengths of output ids                                        |
|                |       `cum_log_probs`        | [batch_size, beam_width]       |   float   |                             cumulative log probability of output sentence (optional)                              |
|   parameter    |                              |                                |           |                                                                                                                   |
|                |      `tensor_para_size`      |                                |    int    |                                      parallelism ways in tensor parallelism                                       |
|                |     `pipeline_para_size`     |                                |    int    |                                     parallelism ways in pipeline parallelism                                      |
|                |         `data_type`          |                                |  string   |                                    data type for inference ("fp32" or "fp16")                                     |
|                |         `model_type`         |                                |  string   |                                                must use `GPT-NeoX`                                                |
|                |   `model_checkpoint_path`    |                                |  string   |                                         the path to save weights of model                                         |
|                |  `enable_custom_all_reduce`  |                                |   bool    |                                          use custom all reduction or not                                          |
| model_transaction_policy |                    |                                |           |                                                                                                                   |
|                |          `decoupled`         |                                |   bool    |              activate the decoupled (streaming) inference, see [#decoupled-mode](#decoupled-mode)                 |

### Decoupled mode

The backend provides a decoupled mode to get intermediate results as soon as they're ready. You can activate this mode by setting the `decoupled` switch to `True`. Then, each time the model has sampled a new token, Triton will send back results. Have a look at the client example in `tools/issue_request.py` to see how you can leverage this feature. You can run a test request with `python3 tools/issue_request.py tools/requests/sample_request_stream.json`.


### Prepare Triton GPT-NeoX model store

Following the guide [#setup](../README.md#setup) to prepare the docker image.

First, download the GPT-NeoX checkpoint:

```shell
docker run -it --rm --gpus=all -v ${WORKSPACE}:${WORKSPACE} -w ${WORKSPACE} ${TRITON_DOCKER_IMAGE} bash
# now in docker

export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
git clone https://github.com/NVIDIA/FasterTransformer.git # Used for converting the checkpoint
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P EleutherAI
export PYTHONPATH=$PWD/FasterTransformer/:$PYTHONPATH
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptneox/utils/eleutherai_gpt_neox_convert.py \
        ${WORKSPACE}/EleutherAI/ \
        ${WORKSPACE}/all_models/gptneox/fastertransformer/1 \
        --tensor-parallelism 2
```

You can set the tensor parallelism to something that fits your needs.

Then we will get the model weights (`xxx.bin`) and the model config file (`config.ini`) in the `${WORKSPACE}/all_models/gptneox/fastertransformer/1/N-gpu/`.

## Run Serving on Single Node

### Run serving directly

```bash
CUDA_VISIBLE_DEVICES=0,1 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=${WORKSPACE}/all_models/gptneox/ &
python3 ${WORKSPACE}/tools/gpt/identity_test.py
```

You can modify `ft_workspace/fastertransformer_backend/tools/identity_test.py` to have different `batch size`, `input length` and `output length` in requests.

* Note: If user encounter `[ERROR] world_size (4) should equal to tensor_para_size_ * pipeline_para_size_ (1 * 1 here)`, please check that the GPU number of your device and set the GPUs you want to use by `CUDA_VISIBLE_DEVICES`. 
* Recommend modifying the SERVER_TIMEOUT of common/util.sh to longer time

#### Run GPT-NeoX end-to-end serving by Triton ensemble

We provide an end-to-end example of GPT-NeoX at `tools/end_to_end_test.py`. Users can run it by 

```bash
python3 ${WORKSPACE}/tools/gpt/end_to_end_test.py
```

after launching the triton server.

Regarding `bad_words_dict` and `stop_words_dict`, they should provide a single CSV-formatted string per item. The string then represents a list of words or expressions and each element is tokenized for further use by the model. Beware of tokenizer subtleties, for example, "word" and " word" are two distinct tokens. You can use the script in `all_models/gpt/preprocessing/1/word_list.py` to help you understand the tokenization.

#### Evaluate the accuracy of GPT-NeoX model on LAMBADA.

```bash
wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -P models
export PYTHONPATH=${WORKSPACE}:$PYTHONPATH
python3 ${WORKSPACE}/tools/gpt/evaluate_lambada.py --datasets_dir models/ --batch_size 16
```

The results would be like

```bash
[INFO] last token accuracy:  xxx% (total token num: 5153)
```
