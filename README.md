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

The Triton backend for the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer). This repository provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. In the FasterTransformer v4.0, it supports multi-gpu inference on GPT-3 model. This backend integrates FasterTransformer into Triton to use giant GPT-3 model serving by Triton. In the below example, we will show how to use the FasterTransformer backend in Triton to run inference on a GPT-3 model with 345M parameters trained by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). In latest beta release, FasterTransformer backend supports the multi-node multi-GPU inference on T5 with the model of huggingface. 

Note that this is a research and prototyping tool, not a formal product or maintained framework. User can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems on the [issues page](https://github.com/triton-inference-server/fastertransformer_backend/issues) in this FasterTransformer_backend repo.

## Table Of Contents
 
- [FasterTransformer Backend](#fastertransformer-backend)
  - [Table Of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Prepare docker images](#prepare-docker-images)
      - [Rebuilding FasterTransformer backend (optional)](#rebuilding-fastertransformer-backend-optional)
    - [How to set the model configuration](#how-to-set-the-model-configuration)
    - [Prepare Triton GPT model store](#prepare-triton-gpt-model-store)
      - [INT8 weight only quantization (**Experimental**)](#int8-weight-only-quantization-experimental)
    - [Prepare Triton T5 model store in the docker](#prepare-triton-t5-model-store-in-the-docker)
  - [NCCL_LAUNCH_MODE](#nccl_launch_mode)
    - [GPUs Topology](#gpus-topology)
  - [MPI Launching with Tensor Parallel size and Pipeline Parallel Size Setting](#mpi-launching-with-tensor-parallel-size-and-pipeline-parallel-size-setting)
  - [Run Serving on Single Node](#run-serving-on-single-node)
    - [Run serving directly](#run-serving-directly)
    - [Benchmark on single node](#benchmark-on-single-node)
  - [Run Triton server on multiple nodes](#run-triton-server-on-multiple-nodes)
    - [Prepare Triton model store for multi-node setup](#prepare-triton-model-store-for-multi-node-setup)
    - [Run on cluster with Enroot/Pyxis support](#run-on-cluster-with-enrootpyxis-support)
    - [How to Run multi-node on the Cluster with Slurm and Docker support](#how-to-run-multi-node-on-the-cluster-with-slurm-and-docker-support)
  - [Changelog](#changelog)

## Introduction

FasterTransformer backend hopes to integrate the FasterTransformer into Triton, leveraging the efficiency of FasterTransformer and serving capabilities of Triton. To run the GPT-3 model, we need to solve the following two issues: 1. How to run the auto-regressive model? 2. How to run the model with multi-gpu and multi-node?

For the issue of auto-regressive model, the workflow of auto-regressive model is like:

1. FasterTransformer backend receives input [A]
2. Compute the query (q), key (k) and value (v) by the input [A].
3. Compute attention: `qk = q * k` and `qkv = qk * v`.
4. Compute other operations of transformer, like Feed Forward Network.
5. Generate next token B, return to Triton server.
6. FasterTransformer backend receives inputs [A, B]
7. Compute the query (q') by [B], keys (k') and values (v') by the inputs [A, B].
8. Compute attention: `qk' = q' * k'` and `qkv' = qk' * v'`.
9. Compute other operations of transformer, like Feed Forward Network.
10. Generate next token C, return to Triton server.

We see that we need to compute the attention by current query and all keys and values. We can find that some computing are wasted, like computing the key of A at step 6 because we have the same results at step 2. To prevent these wasted computing, we need a mechanism to store these states. Currently, Triton does not support such feature, so FasterTransformer handles the whole workflow, storing the keys and values states, and only return the final results. The workflow in FasterTransformer is:

1. Allocate a cache buffer K and V for keys and values respectively.
2. FasterTransformer backend receives input [A]
3. Compute the query (q), key (k) and value (v) by the input [A]. Put the k into K[0] and v into V[0].
4. Compute attention: `qk = q * K[:1]` and `qkv = qk * V[:1]`.
5. Compute other operations of transformer, like Feed Forward Network.
6. Generate next token B, set [B] as input.
7. Compute the query (q'), key (k') and value (v') by the input [B]. Put the k' into K[1] and v' into V[1].
8. Compute attention: `qk' = q' * K[:2]` and `qkv' = qk' * V[:2]`.
9. Compute other operations of transformer, like Feed Forward Network.
10. Generate next token C, set [C] as input.

For the issue of running the model with multi-gpu and multi-node, FasterTransformer backend uses the MPI to communicate between multiple nodes, and uses multi-threads to control the GPUs in one node. Figure 1 demonstrates the workflow of multi-gpu and multi-node in FasterTransformer backend.

<div align=center><img src ="images/multi_gpu_multi_node_workflow.png "/></div>
<div align=center>Fig. 1 Workflow of multi-gpu and multi-node in FasterTransformer backend.</div>

## Setup

```bash
export WORKSPACE=$(pwd)
export SRC_MODELS_DIR=${WORKSPACE}/models
export TRITON_MODELS_STORE=${WORKSPACE}/triton-model-store
export CONTAINER_VERSION=21.07
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
```

### Prepare docker images

The current official Triton Inference Server docker image doesn't contain
FasterTransformer backend, thus the users must prepare own docker image using below command:

```bash
cd ${WORKSPACE}
git clone https://github.com/triton-inference-server/fastertransformer_backend.git -b dev/v1.1_beta
git clone https://github.com/triton-inference-server/server.git # We need some tools when we test this backend
git clone -b dev/v5.0_beta https://github.com/NVIDIA/FasterTransformer # Used for convert the checkpoint and triton output
ln -s server/qa/common .
cd fastertransformer_backend
docker build --rm   \
    --build-arg TRITON_VERSION=${CONTAINER_VERSION}   \
    -t ${TRITON_DOCKER_IMAGE} \
    -f docker/Dockerfile \
    .
```

For testing purposes' docker image will also contain set of tools for model deployment testing.

Push built docker images to docker registry, so that we can later obtain it and initialize it on multiple nodes.

```bash
docker tag ${TRITON_DOCKER_IMAGE} <github_or_gitlab/repo_name/image_name>:${CONTAINER_VERSION}
docker push <github_or_gitlab/repo_name/image_name>:${CONTAINER_VERSION}
```
#### Rebuilding FasterTransformer backend (optional)

Everytime you need to build updated fastertransformer_backend you can build docker image.

But also you can build it manually in interactive session (ex during fixing code on target node) with:

```bash
docker run -it \
    -v ${WORKSPACE}:/workspace \
    --name ft_backend_builder \
    ${TRITON_DOCKER_IMAGE} bash
# in docker container
rm /opt/tritonserver/lib/cmake/FasterTransformer/ -rf # Remove original library
cd fastertransformer_backend
mkdir build -p && cd build && \
    cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install
```

where `${WORKSPACE}` should contain `fastertransformer_backend` directory with code to build.

Then you can commit changes to new docker image with:

```bash
docker commit ft_backend_builder ${TRITON_DOCKER_IMAGE}
```

### How to set the model configuration

In triton backend, the model configuration is controled by the `config.pbtxt` in the folder we launch triton server. For example, the config file of FasterTransformer GPT backend is put in `all_models/gpt/fastertransformer/config.pbtxt`, and the config file of FasterTransformer T5 backend is put in `all_models/t5/fastertransformer/config.pbtxt`.

In the config file of GPT, we can control the model hyper-parameters, like the `head_num` and `num_layer`. We can also control some inference hyper-parameters like `topk` in it.

In the config file of T5, we can only control the inference hyper-parameter like `topk`. The model hyper-parameters are controled by the `config.ini` file which put in `model_checkpoint_path`. If the backend cannot file the model config file, it would crash and give error message.

### Prepare Triton GPT model store

Obtain Megatron 345M model checkpoint:

```shell
cd ${WORKSPACE}
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P models
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ${SRC_MODELS_DIR}/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d models/megatron-models/345m
mkdir ${TRITON_MODELS_STORE}/fastertransformer/1 -p
```

Having Megatron checkpoint, we need to convert to format handled by FasterTransformer. 
If you want to run the model with tensor parallel size 4 and pipeline parallel size 2,
you should convert checkpoints with `-infer_gpu_num = [tensor_para_size], i.e. -infer_gpu_num = 4`. 
We will convert it directly to directory structure which later we'd use as Triton model store.

```shell
cd ${WORKSPACE}
docker run --rm -it --gpus=all \
    -e SRC_MODELS_DIR=${SRC_MODELS_DIR} \
    -e TRITON_MODELS_STORE=${TRITON_MODELS_STORE} \
    -v ${WORKSPACE}:${WORKSPACE} \
    ${TRITON_DOCKER_IMAGE} \
    bash -c "python ${WORKSPACE}/FasterTransformer/examples/pytorch/gpt/utils/megatron_ckpt_convert.py \\
        -i ${SRC_MODELS_DIR}/megatron-models/345m/release/ \\
        -o ${TRITON_MODELS_STORE}/fastertransformer/1 \\
        -trained_gpu_num 1 \\
        -infer_gpu_num 8 \\
        -head_num 16"
```

Copy the sample Triton model configuration with:

```shell
cd ${WORKSPACE}
cp fastertransformer_backend/all_models/gpt/fastertransformer/config.pbtxt ${TRITON_MODELS_STORE}/fastertransformer
```

Modify the Triton model configuration. User can modify the following hyper-parameters:

- `top_k`: k value of top k
- `top_p`: p value of top p
- `tensor_para_size`: size of tensor parallelism
- `layer_para_size`: size of layer parallelism
- `max_seq_len`: max supported sequence length
- `is_half`: Using half or not
- `head_num`: head number of attention
- `size_per_head`: size per head of attention
- `vocab_size`: size of vocabulary
- `decoder_layers`: number of transformer layers
- `max_batch_size`: max supported batch size
- `beam_width`: beam size for beam search
- `temperature`: temperature for logits adjusting
- `repetition_penalty`: repetition penalty for logits adjusting
- `len_penalty`: length penalty for logits adjusting
- `beam_search_diversity_rate`: hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)
- `int8_mode`: 0 means disable. 1 means only quantizing the weight to int8.

#### INT8 weight only quantization (**Experimental**)

To accelerate the inference speed of giant model on small batch size, we add supporting of **INT8 weight only quantization**. Unlike traditional quantization which quantizes inputs, outputs and weights of GEMM, we only quantize the weight here. So, the model can keep the capability without fine-tune. For GEMM computing, the weight sizes are much larger than the size of inputs and outputs, using INT8 weight can reduce the time of loading weights from global memory. For GPT-175B with batch size 1, this brings about 1.3 ~ 1.4x speedup.

However, there are some limitation for this features.

1. The INT8 weight only kernel only brings speedup for batch size <= 2 now.
2. Due to reason 1, we need to maintain both FP16 and INT8 weights at the same time to get better speed. This causes the model memory requirement grows 1.5x.
3. The results of INT8 and FP16 weights may be little different.

### Prepare Triton T5 model store in the docker

Current docker file still not support T5 model directly. Need to install some packages manually.

1. Download the t5-base model checkpoint from [huggingface](https://huggingface.co/models):

```shell
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/t5-base
```

2. Convert the checkpoint.

```shell
pip install -r ../tools/t5_utils/t5_requirement.txt
python _deps/repo-ft-src/examples/pytorch/t5/utils/t5_ckpt_convert.py \
  -o ../all_models/t5/fastertransformer/1/ -i t5-base/ -infer_gpu_num 2
```

Then we will get the model weights (`xxx.bin`) and the config file of model (`config.ini`) in the `../all_models/t5/fastertransformer/1/2-gpu/`. The `config.ini` file contains the hyper-parameters of both encoder and decoder. Other hyper-parameters for inference are put in `../all_models/t5/fastertransformer/config.pbtxt`. User can modify the following hyper-parameters:

- `top_k`: k value of top k
- `top_p`: p value of top p
- `tensor_para_size`: size of tensor parallelism
- `layer_para_size`: size of layer parallelism
- `max_decoding_seq_len`: max supported sequence length in decoding
- `max_encoder_seq_len`: max supported sequence length in encoder
- `is_half`: using half or not
- `max_batch_size`: max supported batch size
- `beam_width`: beam size for beam search
- `temperature`: temperature for logits adjusting
- `repetition_penalty`: repetition penalty for logits adjusting
- `len_penalty`: length penalty for logits adjusting
- `beam_search_diversity_rate`: hyper-parameter for [simple diverse decoding](https://arxiv.org/pdf/1611.08562.pdf)

## NCCL_LAUNCH_MODE

In the docker file, `NCCL_LAUNCH_MODE=GROUP` is the default because it is less likely to hang. However, `NCCL_LAUNCH_MODE=PARALLEL` can bring better performance for 
communication. Hence, users may be able to try to use `NCCL_LAUNCH_MODE=PARALLEL` to accelerate.

In current environment:
```shell
export NCCL_LAUNCH_MODE=PARALLEL
```

When building the Docker container changing the Dockerfile:
```dockerfile
ENV NCCL_LAUNCH_MODE=PARALLEL
```

Or passing environment variable on container start:
```shell
docker run -e NCCL_LAUNCH_MODE=PARALLEL ...
```

### GPUs Topology

If your current machine/nodes are fully connected through PCIE or even across NUMA nodes, there could be poor NCCL performance or even NCCL hangs due to limited peer to peer communication. You can apply `nvidia-smi topo -m` to check the topology.

If you met timed-out or hangs, please first check the topology and try to use DGX V100 or DGX A100 with nvlink connected.

## MPI Launching with Tensor Parallel size and Pipeline Parallel Size Setting

We apply MPI to start single-node/multi-node servers.

- N: Number of MPI Processes/Number of Nodes
- T: Tensor Parallel Size. Default 8
- P: Pipeline Parallel Size. Default 1

`total number of gpus = num_gpus_per_node x N = T x P`

**Note** that we currently do not support the case that different nodes have different number of GPUs.

We start one MPI process per node. If you need to run on three nodes, then you should launch 3 Nodes with one process per node.
Remeber to change `tensor_para_size` and `pipeline_para_size` if you run on multiple nodes. 

We do suggest tensor_para_size = number of gpus in one node (e.g. 8 for DGX A100), and pipeline_para_size = number of nodes (2 for two nodes). Other model configuration in config.pbtxt should be modified as normal.

## Run Serving on Single Node

### Run serving directly

```bash
docker run -it --rm --gpus=all -v ${TRITON_MODELS_STORE}:/model-store:ro -v ${WORKSPACE}:/ft_workspace ${TRITON_DOCKER_IMAGE} bash
# now in docker
mpirun -n 1 --allow-run-as-root /opt/tritonserver/bin/tritonserver --model-repository=/model-store &
bash /ft_workspace/fastertransformer_backend/tools/run_client.sh
python /ft_workspace/FasterTransformer/examples/pytorch/gpt/utils/gpt_token_converter.py \
  --out_file=triton_out \
  --vocab_file=/ft_workspace/models/gpt2-vocab.json \
  --bpe_file=/ft_workspace/models/gpt2-merges.txt
```

* Note: If user encounter `[ERROR] world_size (4) should equal to tensor_para_size_ * pipeline_para_size_ (1 * 1 here)`, please check that the GPU number of your device and set the GPUs you want to use by `CUDA_VISIBLE_DEVICES`. 
* Recommend modifying the SERVER_TIMEOUT of common/util.sh to longer time

### Benchmark on single node

Run this script with different batch size, input_len, output_len, num of runs on a single node with 8 gpus, it will start the server, then start the client to get the latency and stop the server at the end.

**TODO: need to update benchmark script for new Dockerfiles**

```
# run with batch_size = 8, input_len = 512, output_len = 16, and run 10 times to get the average latency
bash $WORKSPACE/fastertransformer_backend/tools/benchmark_single_node.sh -b 8 -i 512 -o 16 -n 10
```

## Run Triton server on multiple nodes

### Prepare Triton model store for multi-node setup

For this experiment you need to [prepare Triton model store](#prepare-triton-model-store):
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
Remeber to change `tensor_para_size` and `pipeline_para_size` as suggested in [MPI Launching with Tensor Parallel size/ Pipeline Parallel Size Setting](#mpi-launching-with-tensor-parallel-size-and-pipeline-parallel-size-setting)  if you run on multiple nodes. 

```bash
WORKSPACE="/workspace" # the dir you build the docker
IMAGE="github_or_gitlab/fastertransformer/multi-node-ft-triton-backend:latest"
CMD="/opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models/gpt"
srun -N 2 -n 2 --mpi=pmix -o inference_server.log --container-mounts /home/account/your_network_shared_space/triton:/workspace --container-name multi-node-ft-triton --container-image $IMAGE bash -c "$CMD"
```

Then, you need to run the server on the background since it will not detach by itself. You can enter and commands `ctrl D` and `bg` or run the script above with `sbatch`.

Next, enter the master triton node (the node where MPI_Rank = 0, normally it is the allocated node with the smallest id) when servers have been started shown in the inference log:

```bash
srun -w master-node-name --overlap --container-name multi-node-ft-triton --container-mounts /home/account/your_network_shared_space/triton:/workspace --pty bash # --overlap may not be needed in your slurm environment
```

Finally, run the client in the master triton node:

```bash
export WORKSPACE="/workspace"
bash $WORKSPACE/fastertransformer_backend/tools/run_client.sh
```

You can refer to `inference_server.log` on the login-node for the inference server log.
When you enter the master triton node, and send a request through the client, you can get the `client.log`, `error.log` and `triton_out` in the current directory.

You can modify `$WORKSPACE/fastertransformer_backend/tools/identity_test.py` to have different `batch size`, `input length` and `output length` in requests.

* Run GPT end-to-end serving by Triton ensemble

We provide an end-to-end example of GPT at `tools/end_to_end_test.py`. Users can run it by 

```bash
python $WORKSPACE/fastertransformer_backend/tools/end_to_end_test.py
```

after launching the triton server.

* Evaluate the accuracy of GPT model on LAMBADA.

```bash
wget https://github.com/cybertronai/bflm/raw/master/lambada_test.jsonl
python $WORKSPACE/fastertransformer_backend/tools/evaluate_lambada.py --n-gram-disable
```

The results would be like

```bash
[INFO] last token accuracy:  xxx% (total token num: 5153)
```

Note that we not only verify the top1 token, but also provide the accuracy of at most 4-gram because 1-gram does not fully verify the correctness of FT. The accuracy of 4-gram is often very slow because it is hard to get fully same results when we compare 4 grams each time, especially when the length of context is short. User can run the evaluation by removing the `--n-gram-disable`.

```bash
python $WORKSPACE/fastertransformer_backend/tools/evaluate_lambada.py
```

* Evaluate the accuracy of T5 model.

We provide a script to evaluate the T5 model by running translation from English to German on the testing dataset of FasterTransformer. Note that we need to launch the t5 backend. We skip the details here. User can run the evaluation by 

```bash
python $WORKSPACE/fastertransformer_backend/tools/t5_utils/t5_end_to_end_test.py
```

The results would be like

```bash
[INFO] ft_triton translates 24 batches taking 8.94 sec to translate 61374 tokens, BLEU score: 27.26, 6862 tokens/sec.
```


### How to Run multi-node on the Cluster with Slurm and Docker support

In order to run multiple nodes, you have to make sure that two nodes can access to each other without ssh issues. The process is almost the same as Enroot/Pyxis clusters: run servers on two nodes with MPIRUN or PMIX, and go to the master node to send requests to servers through the client. The script may differ according to your clusters and environment, but all need to make sure two nodes can get ssh access to each other and call MPIRUN on two nodes.

```bash
export IMAGE="github_or_gitlab/fastertransformer/multi-node-ft-triton-backend:latest" # the image you update in the previous step
export WORKSPACE="/home/name/workspace" # your workspace

srun -N2 -n2 -t 600 --pty bash # Assume the two nodes are luna-01, luna-02

srun -N2 -n2 docker pull $IMAGE

srun -N2 -n2  nvidia-docker run -itd --rm --privileged --network=host --pid=host --cap-add=IPC_LOCK --device=/dev/infiniband -v /$CONT_VOL:$HOST_VOL -v $WORKSPACE:$WORKSPACE -w $WORKSPACE --name ft-backend-test $IMAGE /bin/bash

#set up ssh
srun -N2 -n2  nvidia-docker exec -i --env SLURM_NTASKS --env SLURM_NODEID --env SLURM_PROCID --env SLURM_STEP_NODELIST --env SLURMD_NODENAME --privileged ft-backend-test bash -c "mkdir /root/.ssh && cp $WORKSPACE/ssh/* /root/.ssh && chmod 700 /root/.ssh && chmod 640 /root/.ssh/authorized_keys && chmod 400 /root/.ssh/id_rsa && apt-get update && apt-get install ssh -y && mkdir /run/sshd/ && /usr/sbin/sshd -p 11068 && nvidia-smi -lgc 1530"

# luna-01, luna-02
nvidia-docker exec -ti ft-backend-test bash

cd fastertransformer_backend/build

mpirun --allow-run-as-root -np 2 -H luna-01:1,luna-02:1 -mca plm_rsh_args "-p 11068" /opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models/gpt &

bash $WORKSPACE/fastertransformer_backend/tools/run_client.sh
```

## Changelog

Nov 2021
- Release FasterTransformer backend 1.1 beta version 2.
  - Support Multi-node Multi-GPU T5.
  - Support INT8 weight only quantization (**Experimental**).

Sep 2021
- Release FasterTransformer backend 1.1 beta version 1.
  - Support Multi-node on GPT.

Apr 2021
- **Release the FasterTransformer backend 1.0**.
  - Support Multi-GPU on GPT.