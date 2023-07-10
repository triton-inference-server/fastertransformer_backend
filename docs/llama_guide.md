# LLaMa 

We have deployed LLaMa on triton inference server with faster transformer backend, here is our environment:

+ Ubuntu 20.04
+ docker: 24.0.2
+ cmake
+ python: 3.10.6
+ pip: 23.1.2

Hardware:
+ RTX 3090 (24G VMEM) * 2

follow the tutorial and you will finish the deploying work fine on multi-GPU.

## 0. prepare workspace
The build work requires a new directory.

```bash
cd /to/your/workspace
mkdir ft_triton && cd ft_triton
mkdir llama_deploy && cd llama_deploy
```

We will expand our work in `llama_deploy` directory.

## 1. build docker image
To reproduce all further steps that would be easier to run everything into Docker container. So it's necessary to build a triton docker image for next steps.

The reason why we choose the image tag:23.04 is that this may support the decoupled mode. See this [issue](https://github.com/triton-inference-server/server/issues/6002#issuecomment-1617106369) for more info.

```bash
git clone https://github.com/void-main/fastertransformer_backend.git

cd fastertransformer_backend

sudo docker build --rm --build-arg TRITON_VERSION=23.04 -t triton_ft_backend:23.04 -f docker/Dockerfile .
```

The build process may take more than five minutes, depending on your hardware.

When finished, launch the container:

```bash
cd ../

sudo docker run -it --rm --gpus=all --net=host --shm-size=4G  -v $(pwd):/ft_workspace -p8888:8888 -p8000:8000 -p8001:8001 -p8002:8002 triton_ft_backend:23.04 bash 
```

We have mapped the `llama_deploy` directory to `/ft_workspace` inside the container.

## 2. model convertion and path configuration

We need to prepare the model (converted to Megatron weights binary files) and configure the path in many files to ensure the further steps:

### 2.1 prepare the model

__Assume that you have download the LLaMa-7b-hf (huggingface) model in your computer__

```bash
cd /ft_workspace

git clone https://github.com/void-main/FasterTransformer.git

cd FasterTransformer

sudo mkdir models && sudo chmod -R 777 ./*

python3 ./examples/cpp/llama/huggingface_llama_convert.py -saved_dir=./models/llama -in_file=/your/path/to/llama-7b-hf -infer_gpu_num=2 -weight_data_type=fp16 -model_name=llama
```

**caution:** We are infering llama-7b on 2 GPU, you should change the `-infer_gpu_num` to meet your environment.

### 2.2 configure the in `config.pbtxt`

```bash
cd ..
mkdir triton-model-store
cp -r ./fastertransformer_backend/all_models/llama triton-model-store/
```

You should check the `triton-model-store/llama/fastertransformer/config.pbtxt`:

Change the `tensor_para_size` according to your need. For our two gpu situation, just change it to `2`.

```
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "2"
  }
}
```

Change the `model_checkpoint_path` according to where you have converted your model in **2.1**.

Here we just converted the model to the `FasterTransformer/models` path, notice that the tritonserver will be launched in docker container, so the path should be changed as docker path, too. 

```
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/ft_workspace/FasterTransformer/models/llama/2-gpu/"
  }
}
```

### 2.3 configure the `model.py` 

You should check :
+ `triton-model-store/llama/preprocess/1/model.py` 
+ `triton-model-store/llama/postprocess/1/model.py`:

```python
self.tokenizer = LlamaTokenizer.from_pretrained("/ft_workspace/llama_7b_hf")
```

Make sure the path meet your `llama-7b-hf` model path. This is important to your **preprocess** ans **postprocess** jobs

## 3. Compile the Faster Transformer Library

The Faster Transformer Library stores the core scripts for llama model supporting, so it's necessary to finish this compile work.

Before compilation, we should also check the configuration:

`FasterTransformer/examples/cpp/llama/llala_config.ini`

```
tensor_para_size=2

model_dir=/ft_workspace/FasterTransformer/models/llama/2-gpu/
```

When finished the configuration, compile the library.

```bash
cd /ft_workspace/FasterTransformer

mkdir build && cd build

git submodule init && git submodule update

pip3 install fire jax jaxlib transformers

cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D PYTHON_PATH=/usr/bin/python3 ..

make -j12
```

The compilation may take several minutes.

## 4. Run triton server

```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/tritonserver/bin/tritonserver  --model-repository=./triton-model-store/llama/
```

When you have seen:

```
I0628 02:59:06.157260 11650 server.cc:633] 
+-------------------+---------+--------+
| Model             | Version | Status |
+-------------------+---------+--------+
| ensemble          | 1       | READY  |
| fastertransformer | 1       | READY  |
| postprocessing    | 1       | READY  |
| preprocessing     | 1       | READY  |
+-------------------+---------+--------+

I0628 02:59:06.177118 11650 metrics.cc:864] Collecting metrics for GPU 0: NVIDIA GeForce RTX 3090
I0628 02:59:06.177133 11650 metrics.cc:864] Collecting metrics for GPU 1: NVIDIA GeForce RTX 3090
I0628 02:59:06.177224 11650 metrics.cc:757] Collecting CPU metrics
I0628 02:59:06.177306 11650 tritonserver.cc:2264] 
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| Option                           | Value                                                                                                                      |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------+
| server_id                        | triton                                                                                                                     |
| server_version                   | 2.29.0                                                                                                                     |
| server_extensions                | classification sequence model_repository model_repository(unload_dependents) schedule_policy model_configuration system_sh |
|                                  | ared_memory cuda_shared_memory binary_tensor_data statistics trace logging                                                 |
| model_repository_path[0]         | ./triton-model-store/llama/                                                                                                |
| model_control_mode               | MODE_NONE                                                                                                                  |
| strict_model_config              | 0                                                                                                                          |
| rate_limit                       | OFF                                                                                                                        |
| pinned_memory_pool_byte_size     | 268435456                                                                                                                  |
| cuda_memory_pool_byte_size{0}    | 67108864                                                                                                                   |
| cuda_memory_pool_byte_size{1}    | 67108864                                                                                                                   |
| response_cache_byte_size         | 0                                                                                                                          |
| min_supported_compute_capability | 6.0                                                                                                                        |
| strict_readiness                 | 1                                                                                                                          |
| exit_timeout                     | 30                                                                                                                         |
+----------------------------------+----------------------------------------------------------------------------------------------------------------------------+

I0628 02:59:06.177847 11650 grpc_server.cc:4819] Started GRPCInferenceService at 0.0.0.0:8001
I0628 02:59:06.177982 11650 http_server.cc:3477] Started HTTPService at 0.0.0.0:8000
I0628 02:59:06.219577 11650 http_server.cc:184] Started Metrics Service at 0.0.0.0:8002
```

That means the program was launched successfully.

# Update
+ offer `int8_mode` support in `libfastertransformer.cc` to make sure the compiler can find matching function.
+ fix the `decoupled mode` support, you may get access to decoupled mode with a higher version of tritonserver base image! (23.04 tested)