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

# FasterTransformer GPT Triton Backend

The FasterTransformer GPT implementation are in [gpt_guide.md](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md).

## Table Of Contents

- [FasterTransformer GPT Triton Backend](#fastertransformer-gpt-triton-backend)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup Environment](#setup-environment)
    - [How to set the model configuration](#how-to-set-the-model-configuration)
    - [Decoupled mode](#decoupled-mode)
    - [Prepare Triton GPT model](#prepare-triton-gpt-model)
      - [INT8 weight only quantization (**Experimental**)](#int8-weight-only-quantization-experimental)
  - [Run Serving on Single Node](#run-serving-on-single-node)
    - [Run serving directly](#run-serving-directly)
      - [Run GPT end-to-end serving by Triton ensemble](#run-gpt-end-to-end-serving-by-triton-ensemble)
      - [Evaluate the accuracy of GPT model on LAMBADA.](#evaluate-the-accuracy-of-gpt-model-on-lambada)
      - [Run the Squad Evaluation based on NeMo GPT with prompt learning](#run-the-squad-evaluation-based-on-nemo-gpt-with-prompt-learning)
      - [Issue request directly](#issue-request-directly)
    - [Interactive Text Generation](#interactive-text-generation)
    - [Run XGLM](#run-xglm)
  - [Run Triton server on multiple nodes](#run-triton-server-on-multiple-nodes)
    - [Prepare Triton model store for multi-node setup](#prepare-triton-model-store-for-multi-node-setup)
    - [Run on cluster with Enroot/Pyxis support](#run-on-cluster-with-enrootpyxis-support)

## Introduction

This document describes how to serve the `GPT` model by FasterTransformer Triton backend. This backend is only an interface to call FasterTransformer in Triton. All implementation are in [FasterTransformer repo](https://github.com/NVIDIA/FasterTransformer).

## Setup Environment

Follow the guide in [`README.md`](../README.md) to setup the environment and prepare docker image. We assume users already build the docker here.

### How to set the model configuration

Generally, we need two configuration files to server the FasterTransformer models.

**Model Configuration: config.ini generated during converting the model**

  Normally, this is will be generated automatically when you converting the model checkpoint to FasterTransformer format. However, some configurations (like start_id, end_id) may need to be modified on your own.
  It is because the converter doesn't know anything about the tokenizer if the original checkpoint configurations don't contain such information.

  We provide an example in `all_models/gpt/fastertransformer/1/config.ini`.

  - This should be placed in the same directory of model weights
  - This will be loaded by fastertransformers.
  - This mainly describes the model structure and prompt hyperparameters, start_id, end_id, and so on.

  The following table shows the details of `config.ini`:

  |  Classification  |            Name            | Tensor/Parameter Shape | Data Type |                                                 Description                                                 |
  | :--------------: | :------------------------: | :--------------------- | :-------: | :---------------------------------------------------------------------------------------------------------: |
  |       gpt        |                            |                        |           |                                                                                                             |
  |                  |     `max_pos_seq_len`      |                        |    int    | maximum sequence length supported for position embedding table (only needed by absolute position embedding) |
  |                  |         `head_num`         |                        |    int    |                 the number of head in transformer attention block. A model hyper-parameter                  |
  |                  |      `size_per_head`       |                        |    int    |                the size of each head in transformer attention block. A model hyper-parameter                |
  |                  |        `inter_size`        |                        |    int    |                         the intermediate size of FFN layer. A model hyper-parameter                         |
  |                  |        `vocab_size`        |                        |    int    |                                           the size of vocabulary.                                           |
  |                  |         `start_id`         |                        |    int    |       the id for start token for un-conditional generation task. In GPT-J, it is often same to end_id       |
  |                  |          `end_id`          |                        |    int    |                                  the id for end token for generation task.                                  |
  |                  |        `num_layer`         |                        |    int    |                          the number of transformer layer. A model hyper-parameter                           |
  | weight_data_type |     `weight_data_type`     |                        |    str    |   the weight data type (stored in fastertransformer format), and will be casted when loaded if necessary    |
  | prompt_learning  |                            |                        |           |                                                                                                             |
  |                  |   `prompt_learning_type`   |                        |    int    |        the prompt learning type: [0] no prompt [1] soft prompt [2] prefix_prompt [3] p/prompt tuning        |
  |                  | `prompt_learning_start_id` |                        |    int    | the prompt learning virtual token start id: only used by p/prompt_tuning to check if id is a prompt or not  |
  |      task_i      |                            |                        |           |                           the prompt learning task: task Name id = i (0, 1, ....)                           |
  |                  |        `task_name`         |                        |    str    |                             the task_name used to load specific prompt weights                              |
  |                  |      `prompt_length`       |                        |    int    |                                       the prompt tokens total length                                        |


**Fastertransformer-Triton Serving Configuration: config.pbtxt**

- This will be loaded by triton servers
- This mainly describes the server and fastertransformer inference hyperparameters, like input, output parameters, model type, tensor para size, and so on.

We provide an example in `all_models/gpt/fastertransformer/config.pbtxt`.

The following table shows the details of config.pbtxt:

|      Classification      |              Name               |              Tensor/Parameter Shape              |      Data Type      |                                                                                         Description                                                                                         |
| :----------------------: | :-----------------------------: | :----------------------------------------------: | :-----------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|          input           |                                 |                                                  |                     |                                                                                                                                                                                             |
|                          |           `input_ids`           |          [batch_size, max_input_length]          |       uint32        |                                                                                input ids after tokenization                                                                                 |
|                          |         `input_lengths`         |                   [batch_size]                   |       uint32        |                                                                             real sequence length of each input                                                                              |
|                          |      `request_output_len`       |                   [batch_size]                   |       uint32        |                                                                             how many tokens we want to generate                                                                             |
|                          |         `runtime_top_k`         |                   [batch_size]                   |       uint32        |                                                                         **Optional**. candidate number for sampling                                                                         |
|                          |         `runtime_top_p`         |                   [batch_size]                   |        float        |                                                                       **Optional**. candidate threshold for sampling                                                                        |
|                          |  `beam_search_diversity_rate`   |                   [batch_size]                   |        float        |                                             **Optional**. diversity rate for beam search in this [paper](https://arxiv.org/pdf/1611.08562.pdf)                                              |
|                          |          `temperature`          |                   [batch_size]                   |        float        |                                                                             **Optional**. temperature for logit                                                                             |
|                          |          `len_penalty`          |                   [batch_size]                   |        float        |                                                                           **Optional**. length penalty for logit                                                                            |
|                          |      `repetition_penalty`       |                   [batch_size]                   |        float        |                                                                         **Optional**. repetition penalty for logit                                                                          |
|                          |          `random_seed`          |                   [batch_size]                   |       uint64        |                                                                           **Optional**. random seed for sampling                                                                            |
|                          |      `is_return_log_probs`      |                   [batch_size]                   |        bool         |                                                            **Optional**. flag to return the log probs of generated token or not.                                                            |
|                          | `is_return_context_embeddings`  |                   [batch_size]                   |        bool         |                                                             **Optional**. flag to return the context tokens embeddings or not.                                                              |
|                          |          `beam_width`           |                   [batch_size]                   |       uint32        |                                                           **Optional**. beam size for beam search, using sampling if setting to 1                                                           |
|                          |           `start_id`            |                   [batch_size]                   |       uint32        |                                        **Optional**. the id for start token for un-conditional generation task. In GPT-J, it is often same to end_id                                        |
|                          |            `end_id`             |                   [batch_size]                   |       uint32        |                                        **Optional**. the id for start token for un-conditional generation task. In GPT-J, it is often same to end_id                                        |
|                          |        `bad_words_list`         |          [batch_size, 2, word_list_len]          |        int32        |                                **Optional**. List of tokens (words) to never sample. Should be generated with `all_models/gpt/preprocessing/1/word_list.py`                                 |
|                          |        `stop_words_list`        |          [batch_size, 2, word_list_len]          |        int32        |                               **Optional**. List of tokens (words) that stop sampling. Should be generated with `all_models/gpt/preprocessing/1/word_list.py`                               |
|                          | `prompt_learning_task_name_ids` |                   [batch_size]                   |       uint32        |                                                                  **Optional**. task_name_id for each sequence in one batch                                                                  |
|                          |    `request_prompt_lengths`     |                  [batch_size],                   |       uint32        |                               **Optional**. Length of prefix soft prompt embedding. This describes how many tokens of soft prompt embedding in each sentence.                               |
|                          |   `request_prompt_embedding`    |  [batch_size, max_prompt_length, hidden_units]   | float/half/bfloat16 | **Optional**. FT will concat them with results of embedding lookup kernel. For prefix soft prompt embedding, the type must be float; while for p/prompt tuning, the type is same to weight. |
|                          |      `request_prompt_type`      |                   [batch_size]                   |       uint32        |                                            **Optional**. Prompt type of request. This is necessary when user pass the prompt embedding by input                                             |
|                          |          `top_p_decay`          |                   [batch_size]                   |        float        |                                                                **Optional**. decay values for top_p factual-nucleus sampling                                                                |
|                          |           `top_p_min`           |                   [batch_size]                   |        float        |                                                              **Optional**. min top_p values for top p factual-nucleus sampling                                                              |
|                          |        `top_p_reset_ids`        |                   [batch_size]                   |       uint32        |                                                    **Optional**. reset ids for reseting top_p values for top p factual-nucleus sampling                                                     |
|          output          |                                 |                                                  |                     |                                                                                                                                                                                             |
|                          |          `output_ids`           |           [batch_size, beam_width, -1]           |       uint32        |                                                                              output ids before detokenization                                                                               |
|                          |        `sequence_length`        |             [batch_size, beam_width]             |       uint32        |                                                                            final sequence lengths of output ids                                                                             |
|                          |    `response_input_lengths`     |             [batch_size, beam_width]             |       uint32        |                                                                            final lengths of input ids in the concatenated output ids                                                        |
|                          |         `cum_log_probs`         |             [batch_size, beam_width]             |        float        |                                                                 **Optional**. cumulative log probability of output sentence                                                                 |
|                          |       `output_log_probs`        | [batch_size, beam_width, request_output_seq_len] |        float        |                                                      **Optional**. It records the log probability of logits at each step for sampling.                                                      |
|                          |      `context_embeddings`       |      [batch_size, beam_width, hidden_units]      |        float        |                                                                       **Optional**. Sum of the input token embeddings                                                                       |
|        parameter         |                                 |                                                  |                     |                                                                                                                                                                                             |
|                          |       `tensor_para_size`        |                                                  |         int         |                                                                           parallelism ways in tensor parallelism                                                                            |
|                          |      `pipeline_para_size`       |                                                  |         int         |                                                                          parallelism ways in pipeline parallelism                                                                           |
|                          |          `model_type`           |                                                  |       string        |                                                                                       must use `GPT`                                                                                        |
|                          |     `model_checkpoint_path`     |                                                  |       string        |                                                                     the path to save weights and configuration of model                                                                     |
|                          |           `int8_mode`           |                                                  |         int         |                                                                             int8 weight only quantization mode                                                                              |
|                          |   `enable_custom_all_reduce`    |                                                  |        bool         |                                                                               use custom all reduction or not                                                                               |
| model_transaction_policy |                                 |                                                  |                     |                                                                                                                                                                                             |
|                          |           `decoupled`           |                                                  |        bool         |                                                    activate the decoupled (streaming) inference, see [#decoupled-mode](#decoupled-mode)                                                     |

### Decoupled mode

The backend provides a decoupled mode to get intermediate results as soon as they're ready. You can activate this mode by setting the `decoupled` switch to `True`. Then, each time the model has sampled a new token, Triton will send back results. Have a look at the client example in `tools/issue_request.py` to see how you can leverage this feature. You can run a test request with `python3 tools/issue_request.py tools/requests/sample_request_stream.json`.

### Prepare Triton GPT model

Following the guide [#setup](../README.md#setup) to prepare the docker image.

Download GPT model checkpoint:

```shell
docker run -it --rm --gpus=all --shm-size=1g --ulimit memlock=-1 -v ${WORKSPACE}:${WORKSPACE} -w ${WORKSPACE} ${TRITON_DOCKER_IMAGE} bash
# now in docker

export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
git clone https://gitlab-master.nvidia.com/dl/FasterTransformer/FasterTransformer.git # Used for convert the checkpoint and triton output
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ${SRC_MODELS_DIR}/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d ${SRC_MODELS_DIR}/megatron-models/345m
export PYTHONPATH=$PWD/FasterTransformer/:$PYTHONPATH
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gpt/utils/megatron_ckpt_convert.py \
        -i ${SRC_MODELS_DIR}/megatron-models/345m/release/ \
        -o ${WORKSPACE}/all_models/gpt/fastertransformer/1 \
        --trained-tensor-parallel-size 1 \
        --infer-gpu-num 8 \
        --head-num 16
```

We need to convert to format handled by FasterTransformer.
If you want to run the model with tensor parallel size 8 and pipeline parallel size 2,
you should convert checkpoints with `-infer_gpu_num = [tensor_para_size], i.e. -infer_gpu_num = 8`.
We will convert it directly to directory structure which later we'd use as Triton model store.

Then we will get the model weights (`xxx.bin`) and the config file of model (`config.ini`) in the `${WORKSPACE}/all_models/gpt/fastertransformer/1/8-gpu/`.


#### INT8 weight only quantization (**Experimental**)

To accelerate the inference speed of giant model on small batch size, we add supporting of **INT8 weight only quantization**. Unlike traditional quantization which quantizes inputs, outputs and weights of GEMM, we only quantize the weight here. So, the model can keep the capability without fine-tune. For GEMM computing, the weight sizes are much larger than the size of inputs and outputs, using INT8 weight can reduce the time of loading weights from global memory. For GPT-175B with batch size 1, this brings about 1.3 ~ 1.4x speedup.

However, there are some limitation for this features.

1. The INT8 weight only kernel only brings speedup for batch size <= 2 now.
2. Due to reason 1, we need to maintain both FP16 and INT8 weights at the same time to get better speed. This causes the model memory requirement grows 1.5x.
3. The results of INT8 and FP16 weights may be little different. But the accuracy of real task are on the same level by our experiments and observation.

## Run Serving on Single Node

### Run serving directly

Before launching server, we suggest run the gemm test first like what we mention [here](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#run-gpt). The gemm test program is put at `/workspace/build/fastertransformer_backend/build/bin/gpt_gemm`.

Follow [Prepare Triton GPT model](#prepare-triton-gpt-model) to prepare model, and assume we are in docker now.

Set the `${WORKSPACE}/all_models/gpt/fastertransformer/config.pbtxt` properly, like setting `model_checkpoint_path` to `${WORKSPACE}/all_models/gpt/fastertransformer/1/8-gpu/`.

```bash
/workspace/build/fastertransformer_backend/build/bin/gpt_gemm 8 1 32 16 64 4096 50257 1 1 1
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=${WORKSPACE}/all_models/gpt &
python3 ${WORKSPACE}/tools/gpt/identity_test.py
```

You can modify `fastertransformer_backend/tools/gpt/identity_test.py` to have different `batch size`, `start length` and `output length` in requests. When the `batch size` or `start length` are different to default, remember to add `--random_start_ids` to initialize the start ids. For example

```bash
python3 ${WORKSPACE}/tools/gpt/identity_test.py --batch_size 32 --start_len 40 --output_len 100 --random_start_ids
```

* Note: If user encounter `[ERROR] world_size (4) should equal to tensor_para_size_ * pipeline_para_size_ (1 * 1 here)`, please check that the GPU number of your device and set the GPUs you want to use by `CUDA_VISIBLE_DEVICES`.
* Recommend modifying the SERVER_TIMEOUT of common/util.sh to longer time

#### Run GPT end-to-end serving by Triton ensemble

We provide an end-to-end example of GPT at `fastertransformer_backend/tools/gpt/end_to_end_test.py`. Users can run it by

```bash
python3 ${WORKSPACE}/tools/gpt/end_to_end_test.py
```

after launching the triton server.

Regarding `bad_words_dict` and `stop_words_dict`, they should provide a single CSV-formatted string per item. The string then represents a list of words or expressions and each element is tokenized for further use by the model. Beware of tokenizer subtleties, for example, "word" and " word" are two distinct tokens. You can use the script in `all_models/gpt/preprocessing/1/word_list.py` to help you understand the tokenization.

#### Evaluate the accuracy of GPT model on LAMBADA.

```bash
wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl -P models
export PYTHONPATH=${WORKSPACE}:$PYTHONPATH
python3 ${WORKSPACE}/tools/gpt/evaluate_lambada.py --datasets_dir models/
```

The results would be like

```bash
[INFO] last token accuracy:  xxx% (total token num: 5153)
```

#### Run the Squad Evaluation based on NeMo GPT with prompt learning

[NeMo Prompt Learning Reference](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/prompt_learning.html)

Note that you need to install [NeMo Packages](https://github.com/NVIDIA/NeMo#installation) in order to have dataset and tokenizers.

```bash
python3 tools/gpt_prompt_learning_squad_task_eval.py
```

#### Issue request directly

```bash
python3 ${WORKSPACE}/tools/issue_request.py tools/requests/sample_request_ensemble.json
```

### Interactive Text Generation

This is the the feature that processes requests inside a loop and adds content to the context without recomputing what has already been computed, which means we can save significant computing time with the saved contexts + generated tokens' K/V cache.

For example, there is a request with [*context_1*] which will generate [*generated_tokens_1*]. After that, we may add new contexts [*context_2*] to it, so that the whole context would be [*context_1, generated_tokens_1, context_2*]; based on that, we will generate new tokens [*generated_tokens_2*]. As a result, saving K/V cache for previous contexts and generated tokens would be beneficial for this case.

In order to support it, we will need addditional inputs besides the default ones shown in [How to set the model configuration](#how-to-set-the-model-configuration):

  | Classification |        Name         | Tensor/Parameter Shape | Data Type |                                       Description                                        |                                      Note                                      |
  | :------------: | :-----------------: | :--------------------: | :-------: | :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------: |
  |     input      |                     |                        |           |                                                                                          |                                                                                |
  |                |    `session_len`    |      [batch_size]      |  uint32   |                             max length of the total session                              |                                                                                |
  |                |    `memory_len`     |      [batch_size]      |  uint32   |                        max length of the memory(kv cache) length                         | memory buffer will be a circular buffer, which only store last N tokens' cache |
  |   parameter    |                     |                        |           |                                                                                          |                                                                                |
  |                | `sequence_batching` |                        |           | need direct sequence_batching + stateful backend to support  interactive text generation |                                                                                |

**Note:**

Interactive text generation only works with stateful [backend + sequence batching (direct mode)](https://github.com/triton-inference-server/server/blob/main/docs/architecture.md#direct).
`max_batch_size` is set to 1 because we need to make sure each request redirected to the model instance exclusively. You only need to set `max_sequence_idle_microseconds`, which controls the timeout of the session.

There will be a phesudo batch dimension (always be 1) in order to support interactive text generation. The input shape will change from [batch_size, ....] to [1, batch_size, ...]. As a result, refer to `all_models/gpt-interative-text-generation` and `tools/interactive_text_generation/identity_test.py`. The commonly used `all_models/gpt` would not work for interactive text generation.

**Example:**

We have an example here (`tools/interactive_text_generation/identity_test.py`).

- `session_len` and `memory_len` are the same configurations described above.
- `session_i` [integer, > 0] is used to distingusish different sessions so that the triton server can redirect them to different model instances exclusively.
- `num_session_steps` controls how many steps for the interactive session.

For example, `python ${WORKSPACE}/tools/interactive_text_generation/identity_test.py --session_id 1 --num_session_steps 2`. The workflow would be [*context_1*], [*generated_tokens_1*], [*context_2*], [*generated_tokens_2*]. Contexts come from the requests.

```bash
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=${WORKSPACE}/all_models/gpt-interactive-text-generation/ &
python3 ${WORKSPACE}/tools/interactive_text_generation/identity_test.py --session_id 1 --num_session_steps 2
```

### Run XGLM

<!-- ```bash
wget https://dl.fbaipublicfiles.com/fairseq/models/xglm/xglm.564M.tar.gz
tar -zxvf  xglm.564M.tar.gz
pip install fairseq
PYTHONPATH=$PWD/_deps/repo-ft-src python3 _deps/repo-ft-src/examples/pytorch/gpt/utils/xglm_convert.py -o ./tmp/ -i ./564M/ -i_g 1
// change `model_checkpoint_path` of `config.pbtxt` to `./tmp/1-gpu/`.
``` -->

```bash
git lfs clone https://huggingface.co/facebook/xglm-564M
PYTHONPATH=$PWD/../ python3 ../examples/pytorch/gpt/utils/huggingface_xglm_convert.py -o /data/hf/tmp/  -i ./xglm-564M/ -i_g 1
# Note that change `model_checkpoint_path` of `config.pbtxt` to `./tmp/1-gpu/`.
```

### Run BLOOM
We provide an end-to-end example of BLOOM at `all_models/bloom` and `tools/gpt/bloom_test.py`, which are almost the same with GPT. A user can prepare a pretrained checkpoint of BLOOM by
```bash
git lfs clone https://huggingface.co/bigscience/bloom-560m
python3 {FT_DIR}/examples/pytorch/gpt/utils/huggingface_bloom_convert.py -o /data/hf/tmp/ -i ./bloom-560m/ -tp 1
# Note that change `model_checkpoint_path` of `config.pbtxt` to `/data/hf/tmp/1-gpu/`.
```
and then similar to GPT we can run an example by
```bash
python3 ${WORKSPACE}/tools/gpt/end_to_end_test.py --model-variant bloom
```

## Run Triton server on multiple nodes

### Prepare Triton model store for multi-node setup

For this experiment you need to [prepare Triton GPT model](#prepare-triton-gpt-model):
- properly convert Megatron checkpoint to FasterTransformer format
- update Triton model configuration

We do suggest:
- `tensor_para_size` = number of gpus in one node (e.g. 8 for DGX A100)
- `layer_para_size` = number of nodes

Other Triton model configuration parameters should be updated as for single node setup.

Model store should be placed on network file system available for all cluster nodes on which Triton will run.

### Run on cluster with Enroot/Pyxis support

First allocate two nodes:

```bash
salloc -A account_name -t 10:00:00 -N 2
```

Then run the script shown below to start two nodes' server.
-N and -n should be equal to the number of nodes because we start one process per node. If you need to run on two nodes, then -N 2 and -n 2.
Remember to change `tensor_para_size` and `pipeline_para_size` as suggested in [MPI Launching with Tensor Parallel size/ Pipeline Parallel Size Setting](#mpi-launching-with-tensor-parallel-size-and-pipeline-parallel-size-setting)  if you run on multiple nodes.

```bash
WORKSPACE="/workspace" # the dir you build the docker
IMAGE="github_or_gitlab/fastertransformer/multi-node-ft-triton-backend:latest"
CMD="/opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models/gpt"
srun -N 2 -n 2 --mpi=pmix -o inference_server.log \
               --container-mounts /home/account/your_network_shared_space/triton:/workspace \
               --container-name multi-node-ft-triton \
               --container-image $IMAGE \
               bash -c "$CMD"
```

Then, you need to run the server on the background since it will not detach by itself. You can enter and commands `ctrl D` and `bg` or run the script above with `sbatch`.

Next, enter the master triton node (the node where MPI_Rank = 0, normally it is the allocated node with the smallest id) when servers have been started shown in the inference log:

```bash
srun -w master-node-name --overlap --container-name multi-node-ft-triton --container-mounts /home/account/your_network_shared_space/triton:/workspace --pty bash # --overlap may not be needed in your slurm environment
```

Finally, run the client in the master triton node:

```bash
python3 fastertransformer_backend/tools/gpt/end_to_end_test.py
```

You can refer to `inference_server.log` on the login-node for the inference server log.
