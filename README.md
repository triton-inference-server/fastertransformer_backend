<!--
# Copyright 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

**Note: FasterTransformer development has transitioned to TensorRT-LLM. All developers are encouraged to leverage [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/release/0.5.0) and [tensorrtllm_backend](https://github.com/triton-inference-server/tensorrtllm_backend) to get the latest improvements on LLM Inference. The NVIDIA/FasterTransformer repo will stay up, but will not have further development.** 

# FasterTransformer Backend

The Triton backend for the [FasterTransformer](https://github.com/NVIDIA/FasterTransformer). This repository provides a script and recipe to run the highly optimized transformer-based encoder and decoder component, and it is tested and maintained by NVIDIA. In the FasterTransformer v4.0, it supports multi-gpu inference on GPT-3 model. This backend integrates FasterTransformer into Triton to use giant GPT-3 model serving by Triton. In the below example, we will show how to use the FasterTransformer backend in Triton to run inference on a GPT-3 model with 345M parameters trained by [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). In latest release, FasterTransformer backend supports the multi-node multi-GPU inference on T5 with the model of huggingface. 

Note that this is a research and prototyping tool, not a formal product or maintained framework. User can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems on the [issues page](https://github.com/triton-inference-server/fastertransformer_backend/issues) in this FasterTransformer_backend repo.

## Table Of Contents
 
- [FasterTransformer Backend](#fastertransformer-backend)
  - [Table Of Contents](#table-of-contents)
  - [Support matrix](#support-matrix)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Prepare docker images](#prepare-docker-images)
      - [Rebuilding FasterTransformer backend (optional)](#rebuilding-fastertransformer-backend-optional)
  - [NCCL\_LAUNCH\_MODE](#nccl_launch_mode)
    - [GPUs Topology](#gpus-topology)
  - [Model-Parallism and Triton-Multiple-Model-Instances](#model-parallism-and-triton-multiple-model-instances)
    - [Run inter-node (T x P \> GPUs per Node) models](#run-inter-node-t-x-p--gpus-per-node-models)
    - [Run intra-node (T x P \<= GPUs per Node) models](#run-intra-node-t-x-p--gpus-per-node-models)
    - [Specify Multiple Model Instances](#specify-multiple-model-instances)
    - [Multi-Node Inference](#multi-node-inference)
  - [Request examples](#request-examples)
  - [Changelog](#changelog)

## Support matrix

| Models   | FP16 | BF16 | Tensor parallel | Pipeline parallel |
| -------- | ---- | ---- | --------------- | ----------------- |
| GPT/OPT  | Yes  | Yes  | Yes             | Yes               |
| BLOOM    | Yes  | Yes  | Yes             | Yes               |
| GPT-J    | Yes  | Yes  | Yes             | Yes               |
| T5/UL2   | Yes  | Yes  | Yes             | Yes               |
| GPT-NeoX | Yes  | Yes  | Yes             | Yes               |
| BERT     | Yes  | Yes  | Yes             | Yes               |

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
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
cd fastertransformer_backend
export WORKSPACE=$(pwd)
export CONTAINER_VERSION=23.04
export TRITON_DOCKER_IMAGE=triton_with_ft:${CONTAINER_VERSION}
```

### Prepare docker images

The current official Triton Inference Server docker image doesn't contain
FasterTransformer backend, thus the users must prepare own docker image either by:
1. Using the build script
   Note the `--is-multistage-build` is optional. It installs the minimal dependencies that allow fastertransformer_backend to run
```bash
# Create your own Triton container. You can skip this step (done in trtionserver/server)
python3 compose.py --backend pytorch --container-version 23.04 --output-name tritonserver_pytorch_only
# In tritonserver/fastertransformer_backend. This will overwrite the current Dockerfile
python3 docker/create_dockerfile_and_build.py --base-image tritonserver_pytorch_only --image-name tritonserver_with_ft --is-multistage-build

```
  Alternatively you can simply run
```bash
python3 create_dockerfile_and_build.py --triton-version 23.04
```
to generate a fastertransformer backend, like done in option 2.

2. Using below command:

```bash
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

Every time you need to build updated fastertransformer_backend you can build docker image.

But also you can build it manually in interactive session (ex during fixing code on target node) with:

```bash
docker run -it \
    --shm-size=1g --ulimit memlock=-1 \
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


## Model-Parallism and Triton-Multiple-Model-Instances
We apply MPI to start single-node/multi-node servers.

- N: Number of MPI Processes/Number of Nodes
- T: Tensor Parallel Size. Default 1
- P: Pipeline Parallel Size. Default 1

Multiple model instances on same GPUs will share the weights, so there will not be any redundant weights memory allocated.

### Run inter-node (T x P > GPUs per Node) models
  - `total number of GPUs = num_gpus_per_node x N = T x P`.
  - only single mode instance supported

### Run intra-node (T x P <= GPUs per Node) models
  - `total number of visible GPUs must be evenly divisble by T x P`. Note that you can control this by setting `CUDA_VISIBLE_DEVICES`.
  - `total number of visible GPUs must be <= T x P x Instance Count`. It can avoid unnecessary cuda memory allocation on unused GPUs.
  - multiple model instances can be run on tsame GPU groups or different GPU groups. 

    The backend will first try to assign different GPU groups to different model instances. If there are not empty GPUs, multiple model instances will be assigned to the same GPU groups.

    For example, if there are 8 GPUs, 8 model instances (T = 2, P = 1), then model instances will be distributed to GPU groups [0, 1], [2, 3], [4, 5], [6, 7], [0, 1], [2, 3], [4, 5], [6, 7]. 
  -  weights are shared among model instances in same GPU groups. In the example above, instance 0 and instance 4 will share the same weights, and others are similar.

### Specify Multiple Model Instances

Set `count` here to start multiple model instances. Note `KIND_CPU` is the only choice here as the backend needs to take full control of how to distribute multiple model instances to all the visible GPUs.

```pbtxt
instance_group [
  {
    count: 8
    kind: KIND_CPU
  }
]
```

### Multi-Node Inference

We currently do not support the case that different nodes have different number of GPUs.

We start one MPI process per node. If you need to run on three nodes, then you should launch 3 Nodes with one process per node.
Remember to change `tensor_para_size` and `pipeline_para_size` if you run on multiple nodes. 

We do suggest tensor_para_size = number of GPUs in one node (e.g. 8 for DGX A100), and pipeline_para_size = number of nodes (2 for two nodes). Other model configuration in config.pbtxt should be modified as normal.

## Request examples

The `tools` directory provides python scripts to send requests to the triton server. You can build upon those examples to suit your needs.

Specifically `tools/issue_request.py` is a simple script that sends a request contained in a JSON file. You may use it with `$python3 tools/issue_request.py tools/requests/sample_request.json`, for example. You can also pass command-line arguments as a JSON-formatted string with the `--params` argument.

## Changelog

Oct 2022
- Support IA3 in T5 and T5-Encoder

Sep 2022
- Support T5-Encoder only backend
- Support T5 prompt tuning and p tuning
- Support factual-nucleus sampling ([link](https://arxiv.org/pdf/2206.04624.pdf))

Aug 2022
- **Release the FasterTransformer backend 1.2**.
- Support for interactive generation

July 2022
- Support shared context optimization in GPT model
- Support UL2

June 2022
- Support decoupled (streaming) mode.
- Add demo of grpc protocol.
- Support BERT

May 2022
- Support GPT-NeoX.
- Support optional input. (triton version must be after 22.05)

April 2022
- **Release the FasterTransformer backend 1.1**.
- Support bfloat16 inference in GPT model.
- Support Nemo Megatron T5 and Megatron-LM T5 model.
- Support optional input in fastertransformer backends. (Only supported after Triton 22.01)

Jan 2022
- Move runtime argument like topk into backend input.

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
