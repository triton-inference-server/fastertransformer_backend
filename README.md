<!--
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

# FasterTransformer Backend

The Triton backend for the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer). This repository provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. In the FasterTransformer v4.0, it supports multi-gpu inference on GPT-3 model. This backend integrates FasterTransformer into Triton to use giant GPT-3 model serving by Triton. In the below example, we will show how to use the FasterTransformer backend in Triton to run inference on a GPT-3 model with 345M parameters trained by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

Note that this is a research and prototyping tool, not a formal product or maintained framework. User can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems on the issues page in this FasterTransformer_backend repo.

<!-- TODO Add the FasterTransformer_backend issue link -->

## Table Of Contents

- [FasterTransformer Backend](#fastertransformer-backend)
  - [Table Of Contents](#table-of-contents)
  - [Setup](#setup)
  - [Run Serving](#run-serving)

## Setup

* Prepare Machine

We provide a docker file, which bases on Triton image `nvcr.io/nvidia/tritonserver:21.02-py3`, to setup the environment.

```bash
mkdir workspace && cd workspace 
git clone https://gitlab-master.nvidia.com/liweim/transformer_backend.git
nvidia-docker build --tag ft_backend --file transformer_backend/Dockerfile .
nvidia-docker run --gpus=all -it --rm --volume $HOME:$HOME --volume $PWD:$PWD -w $PWD --name ft-work  ft_backend
cd workspace
export WORKSPACE=$(pwd)
```

* Install libraries for Megatron (option)

```bash
pip install torch regex fire
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

* Build FT backend

```bash
cd $WORKSPACE
git clone https://github.com/triton-inference-server/server.git
export PATH=/usr/local/mpi/bin:$PATH
source transformer_backend/build.env
mkdir -p transformer_backend/build && cd $WORKSPACE/transformer_backend/build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .. && make -j32
```

* Prepare model

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d models/megatron-models/345m
python ../sample/pytorch/utils/megatron_ckpt_convert.py -i ./models/megatron-models/345m/release/ -o ./models/megatron-models/c-model/345m/ -t_g 1 -i_g 8
python _deps/repo-ft-src/sample/pytorch/utils/megatron_ckpt_convert.py -i ./models/megatron-models/345m/release/ -o ./models/megatron-models/c-model/345m/ -t_g 1 -i_g 8
cp ./models/megatron-models/c-model/345m/8-gpu $WORKSPACE/transformer_backend/all_models/transformer/1/ -r
```

## Run Serving

* Run servning directly

```bash
cp $WORKSPACE/transformer_backend/build/libtriton_transformer.so $WORKSPACE/transformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/transformer
cd $WORKSPACE && ln -s server/qa/common .
# Recommend to modify the SERVER_TIMEOUT of common/utils.sh to longer time
cd $WORKSPACE/transformer_backend/build/
bash $WORKSPACE/transformer_backend/tools/run_server.sh
bash $WORKSPACE/transformer_backend/tools/run_client.sh
python _deps/repo-ft-src/sample/pytorch/utils/convert_gpt_token.py --out_file=triton_out # Used for checking result
```

* Modify the model configuration

The model configuration for Triton server is put in `all_models/transformer/config.pbtxt`. User can modify the following hyper-parameters:

- candidate_num: k value of top k
- probability_threshold: p value of top p
- tensor_para_size: size of tensor parallelism
- layer_para_size: size of layer parallelism
- layer_para_batch_size: Useless in Triton backend becuase this backend only supports single node, and user are recommended to use tensor parallel in single node
- max_seq_len: max supported sequence length
- is_half: Using half or not
- head_num: head number of attention
- size_per_head: size per head of attention
- vocab_size: size of vocabulary
- decoder_layers: number of transformer layers
- batch_size: max supported batch size
- is_fuse_QKV: fusing QKV in one matrix multiplication or not. It also depends on the weights of QKV.