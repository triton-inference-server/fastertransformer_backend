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

Note that this is a research and prototyping tool, not a formal product or maintained framework. User can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems on the [issues page](https://github.com/triton-inference-server/fastertransformer_backend/issues) in this FasterTransformer_backend repo.

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
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
cd fastertransformer_backend
git checkout v1.1-dev
cd ../
nvidia-docker build --tag ft_backend --file fastertransformer_backend/Dockerfile .
nvidia-docker run --gpus=all -it --rm --volume $PWD:/workspace -w /workspace --name ft-work  ft_backend
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
source fastertransformer_backend/build.env
mkdir -p fastertransformer_backend/build && cd $WORKSPACE/fastertransformer_backend/build
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
cp ./models/megatron-models/c-model/345m/8-gpu $WORKSPACE/fastertransformer_backend/all_models/transformer/1/ -r
```

* **Prepare the ft-triton-backend docker**

Push a ft-triton-backend-docker so that we can initilize them on multiple nodes

```bash
ctrl p + ctrl q #detach a container
docker ps -a #get the container name
docker commit container_name github_or_gitlab/repo_name/image_name:latest
docker push github_or_gitlab/repo_name/image_name:latest
```


## Run Serving on Single Node

* Run servning directly

```bash
cp $WORKSPACE/fastertransformer_backend/build/libtriton_transformer.so $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/transformer
cd $WORKSPACE && ln -s server/qa/common .
# Recommend to modify the SERVER_TIMEOUT of common/util.sh to longer time
cd $WORKSPACE/fastertransformer_backend/build/
bash $WORKSPACE/fastertransformer_backend/tools/run_server.sh
bash $WORKSPACE/fastertransformer_backend/tools/run_client.sh
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

## How to Run multi-node on the Cluster with Enroot/Pyxis support

**Warp up everything in a docker**: as described in *Prepare the ft-triton-backend docker* step.

First allocate two nodes:

```bash
salloc -A account_name -t 10:00:00 -N 2
```

Then run the script shown below to start two nodes' server. `Ctrl+Z` and `bg` in order to run on the background.
-N and -n should be equal to the number of nodes because we start one process per node. If you need to run on three nodes, then -N 3 and -n 3.
Remeber to change `tensor_para_size` and `layer_para_size` if you run on multiple nodes (`total number of gpus = num_gpus_per_node x num_nodes = tensor_para_size x layer_para_size`), we do suggest tensor_para_size = number of gpus in one node (e.g. 8 for DGX A100), and layer_para_size = number of nodes (2 for two nodes). Other model configuration in config.pbtxt should be modified as normal.

```bash
WORKSPACE="/workspace" # the dir you build the docker
IMAGE="github_or_gitlab/fastertransformer/multi-node-ft-triton-backend:latest"
CMD="cp $WORKSPACE/fastertransformer_backend/build/libtriton_fastertransformer.so $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/fastertransformer;/opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models"
srun -N 2 -n 2 --mpi=pmix -o inference_server.log --container-mounts /home/account/your_network_shared_space/triton:/workspace --container-name multi-node-ft-triton --container-image $IMAGE bash -c "$CMD"
```

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

## How to Run multi-node on the Cluster with Slurm and Docker support

In order to run multiple nodes, you have to make sure that two nodes can access to each other without ssh issues. The process is almost the same as Enroot/Pyxis clusters: run servers on two nodes with MPIRUN or PMIX, and go to the master node to send requests to servers through the client. The script may differ according to your clusters and environment, but all need to make sure two nodes can get ssh access to each other and call MPIRUN on two nodes.

```bash
export IMAGE="github_or_gitlab/fastertransformer/multi-node-ft-triton-backend:latest" # the image you update in the previous step
export WORKSPACE="/home/name/workspace" # your workspace

srun -N2 -n2 -t 600 --pty bash # Assume the two nodes are luna-01, luna-02

srun -N2 -n2 docker pull $IMAGE

srun -N2 -n2  nvidia-docker run -itd --rm --privileged --network=host --pid=host --cap-add=IPC_LOCK --device=/dev/infiniband -v /$CONT_VOL:$HOST_VOL -w $WORKSPACE --name ft-backend-test $IMAGE /bin/bash

#set up ssh
srun -N2 -n2  nvidia-docker exec -i --env SLURM_NTASKS --env SLURM_NODEID --env SLURM_PROCID --env SLURM_STEP_NODELIST --env SLURMD_NODENAME --privileged ft-backend-test bash -c "mkdir /root/.ssh && cp $PWD/.ssh/* /root/.ssh && chmod 700 /root/.ssh && chmod 640 /root/.ssh/authorized_keys && chmod 400 /root/.ssh/id_rsa && apt-get update && apt-get install ssh -y && mkdir /run/sshd/ && /usr/sbin/sshd -p 11068 && nvidia-smi -lgc 1530"

# luna-01, luna-02
nvidia-docker exec -ti ft-backend-test bash

cd fastertransformer_backend/build

mpirun --allow-run-as-root -np 2 -H luna-01:1,luna-02:1 -mca plm_rsh_args "-p 11068" cp $WORKSPACE/fastertransformer_backend/build/libtriton_transformer.so $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so /opt/tritonserver/backends/transformer

mpirun --allow-run-as-root -np 2 -H luna-01:1,luna-02:1 -mca plm_rsh_args "-p 11068" /opt/tritonserver/bin/tritonserver --model-repository=$WORKSPACE/fastertransformer_backend/all_models &

bash $WORKSPACE/fastertransformer_backend/tools/run_client.sh
```


