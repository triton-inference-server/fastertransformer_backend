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

# FasterTransformer GPT-J Triton Backend

The FasterTransformer GPT-J implementation are in [gptj_guide.md](https://github.com/NVIDIA/FasterTransformer/blob/dev/v5.0_beta/docs/gptj_guide.md).

## Table Of Contents
 
- [FasterTransformer GPT-J Triton Backend](#fastertransformer-gpt-j-triton-backend)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup Environment](#setup-environment)
    - [How to set the model configuration](#how-to-set-the-model-configuration)
    - [Prepare Triton GPT-J model store](#prepare-triton-gpt-j-model-store)
  - [Run Serving on Single Node](#run-serving-on-single-node)
    - [Run serving directly](#run-serving-directly)
      - [Run GPT-J end-to-end serving by Triton ensemble](#run-gpt-j-end-to-end-serving-by-triton-ensemble)
      - [Evaluate the accuracy of GPT-J model on LAMBADA.](#evaluate-the-accuracy-of-gpt-j-model-on-lambada)
  - [Run Triton server on multiple nodes](#run-triton-server-on-multiple-nodes)
    - [Prepare Triton model store for multi-node setup](#prepare-triton-model-store-for-multi-node-setup)
    - [Run on cluster with Enroot/Pyxis support](#run-on-cluster-with-enrootpyxis-support)

## Introduction

This document describes how to serve the `GPT-J` model by FasterTransformer Triton backend. This backend is only an interface to call FasterTransformer in Triton. All implementation are in [FasterTransformer repo](https://github.com/NVIDIA/FasterTransformer). 

## Setup Environment

Follow the guide in [`README.md`](../README.md) to setup the environment and prepare docker image. We assume users already build the docker here.

### How to set the model configuration

In GPT-J triton backend, the serving configuration is controlled by `config.pbtxt`. We provide an example in `all_models/gptj/fastertransformer/config.pbtxt`. It contains the input parameters, output parameters, model configuration, some other settings like `tensor_para_size` and `model_checkpoint_path`.

The following table shows the details of these settings:

* Settings in config.pbtxt

| Classification |             Name             | Tensor/Parameter Shape                     | Data Type |                                           Description                                           |
| :------------: | :--------------------------: | :----------------------------------------- | :-------: | :---------------------------------------------------------------------------------------------: |
|     input      |                              |                                            |           |                                                                                                 |
|                |         `input_ids`          | [batch_size, max_input_length]             |  uint32   |                                  input ids after tokenization                                   |
|                |       `input_lengths`        | [batch_size]                               |  uint32   |                               real sequence length of each input                                |
|                |     `request_output_len`     | [batch_size]                               |  uint32   |                               how many tokens we want to generate                               |
|                |       `runtime_top_k`        | [batch_size]                               |  uint32   |                                  candidate number for sampling                                  |
|                |       `runtime_top_p`        | [batch_size]                               |   float   |                                candidate threshould for sampling                                |
|                | `beam_search_diversity_rate` | [batch_size]                               |   float   |      diversity rate for beam search in this [paper](https://arxiv.org/pdf/1611.08562.pdf)       |
|                |        `temperature`         | [batch_size]                               |   float   |                                      temperature for logit                                      |
|                |        `len_penalty`         | [batch_size]                               |   float   |                                    length penalty for logit                                     |
|                |     `repetition_penalty`     | [batch_size]                               |   float   |                                  repetition penalty for logit                                   |
|                |        `random_seed`         | [batch_size]                               |   float   |                                    random seed for sampling                                     |
|                |    `is_return_log_probs`     | [batch_size]                               |   bool    |                     flag to return the log probs of generated token or not.                     |
|                |         `beam_width`         | [batch_size]                               |  uint32   |                    beam size for beam search, using sampling if setting to 1                    |
|                |       `bad_words_list`       | [batch_size, 2, word_list_len]             |   int32   | List of tokens (words) to never sample. Should be generated with `all_models/gpt/preprocessing/1/word_list.py` |
|                |       `stop_words_list`      | [batch_size, 2, word_list_len]             |   int32   | List of tokens (words) that stop sampling. Should be generated with `all_models/gpt/preprocessing/1/word_list.py` |
|     output     |                              |                                            |           |                                                                                                 |
|                |         `output_ids`         | [batch_size, beam_width, max_input_length] |  uint32   |                                output ids before detokenization                                 |
|                |      `sequence_length`       | [batch_size, beam_width]                   |  uint32   |                              final sequence lengths of output ids                               |
|                |       `cum_log_probs`        | [batch_size, beam_width]                   |  uint32   |                          cumulative log probability of output sentence                          |
|   parameter    |                              |                                            |           |                                                                                                 |
|                |      `tensor_para_size`      |                                            |    int    |                             parallelism ways in tensor parallelism                              |
|                |     `pipeline_para_size`     |                                            |    int    |                            parallelism ways in pipeline parallelism                             |
|                |        `max_seq_len`         |                                            |    int    |                 maximum sequence length supported for position embedding table                  |
|                |          `is_half`           |                                            |   bool    |           using half for inference or not. 0 means to use float, 1 means to use half            |
|                |          `head_num`          |                                            |    int    |           the number of head in transformer attention block. A model hyper-parameter            |
|                |       `size_per_head`        |                                            |    int    |          the size of each head in transformer attention block. A model hyper-parameter          |
|                |         `inter_size`         |                                            |    int    |                   the intermediate size of FFN layer. A model hyper-parameter                   |
|                |         `vocab_size`         |                                            |    int    |                                     the size of vocabulary.                                     |
|                |          `start_id`          |                                            |    int    | the id for start token for un-conditional generation task. In GPT-J, it is often same to end_id |
|                |       `decoder_layers`       |                                            |    int    |                    the number of transformer layer. A model hyper-parameter                     |
|                |      `rotary_embedding`      |                                            |    int    |                         rotary embedding size. A model hyper-parameter                          |
|                |         `model_type`         |                                            |  string   |                                        must use `GPT-J`                                         |
|                |   `model_checkpoint_path`    |                                            |  string   |                                the path to save weights of model                                |
|                |  `enable_custom_all_reduce`  |                                            |   bool    |                                 use custom all reduction or not                                 |

### Prepare Triton GPT-J model store

Download GPT-J model checkpoint:

```shell
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
cd ${WORKSPACE}
git clone https://github.com/NVIDIA/FasterTransformer.git # Used for convert the checkpoint and triton output
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
wget https://mystic.the-eye.eu/public/AI/GPT-J-6B/step_383500_slim.tar.zstd
tar -axf step_383500_slim.tar.gz -C ${SRC_MODELS_DIR}
mkdir ${TRITON_MODELS_STORE}/fastertransformer/1 -p
python3 ${WORKSPACE}/FasterTransformer/examples/pytorch/gptj/utils/gptj_ckpt_convert.py \
        --output-dir ${TRITON_MODELS_STORE}/fastertransformer/1 \
        --ckpt-dir ./${WORKSPACE}/FasterTransformer/step_383500/ \
        --n-inference-gpus 8
```

We need to convert to format handled by FasterTransformer. 
If you want to run the model with tensor parallel size 8 and pipeline parallel size 2,
you should convert checkpoints with `--n-inference-gpus = [tensor_para_size], i.e. --n-inference-gpus = 8`. 
We will convert it directly to directory structure which later we'd use as Triton model store.

Then we will get the model weights (`xxx.bin`) and the config file of model (`config.ini`) in the `${TRITON_MODELS_STORE}/fastertransformer/1/8-gpu/`. 

## Run Serving on Single Node

### Run serving directly

```bash
docker run -it --rm --gpus=all -v ${WORKSPACE}:/ft_workspace ${TRITON_DOCKER_IMAGE} bash
# now in docker
CUDA_VISIBLE_DEVICES=0 mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver  --model-repository=$PWD/../all_models/gptj/ &
python3 ft_workspace/fastertransformer_backend/tools/identity_test.py
```

You can modify `ft_workspace/fastertransformer_backend/tools/identity_test.py` to have different `batch size`, `input length` and `output length` in requests.

* Note: If user encounter `[ERROR] world_size (4) should equal to tensor_para_size_ * pipeline_para_size_ (1 * 1 here)`, please check that the GPU number of your device and set the GPUs you want to use by `CUDA_VISIBLE_DEVICES`. 
* Recommend modifying the SERVER_TIMEOUT of common/util.sh to longer time

#### Run GPT-J end-to-end serving by Triton ensemble

We provide an end-to-end example of GPT-J at `tools/end_to_end_test.py`. Users can run it by 

```bash
python ft_workspace/fastertransformer_backend/tools/end_to_end_test.py
```

after launching the triton server.

Regarding `bad_words_dict` and `stop_words_dict`, they should provide a single CSV-formatted string per item. The string then represent a list of words or expressions and each element is tokenized for further use by the model. Beware of tokenizer subtleties, for example, "word" and " word" are two distinct tokens. You can use the script in `all_models/gpt/preprocessing/1/word_list.py` to help you understand the tokenization.

#### Evaluate the accuracy of GPT-J model on LAMBADA.

```bash
wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl
python ft_workspace/fastertransformer_backend/tools/evaluate_lambada.py --n-gram-disable
```

The results would be like

```bash
[INFO] last token accuracy:  xxx% (total token num: 5153)
```

## Run Triton server on multiple nodes

### Prepare Triton model store for multi-node setup

For this experiment you need to [prepare Triton GPT-J model store](#prepare-triton-gpt-j-model-store):
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
CMD="/opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models/gptj"
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
python3 fastertransformer_backend/tools/end_to_end_test.py
```

You can refer to `inference_server.log` on the login-node for the inference server log.
