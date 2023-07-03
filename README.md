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

**NOTE: Fastertransformer backend is currently undergoing restructuring. Build instructions are only tested with Triton container versions <= `23.04`**. 

# 1. FasterTransformer Backend

**The triton faster transformer backend works as an interface to call FasterTransformer in triton.**

All necessary implements are actually in [FasterTransformer](https://github.com/void-main/FasterTransformer) repository.

The `CMakeList.txt` will fetch relative repository to organize and compile the project with:

+ this repository itself
+ Faster Transformer repository
+ 3rdparty
  + cutlass
  + Megatron
  + etc...

# 2. LLaMa support

To check how faster transformer support LLaMa, and how triton support LLaMa, here is the  structure:

```
Faster Transformer Library
├── examples
│   └── cpp
│       └── llama
│           ├── CMakeList.txt
│           ├── llama_config.ini
│           ├── llama_example.cc
│           └── llama_triton_example.cc
└── src
    └── fastertransformer
        ├── models
        │   └── llama
        │       ├── CMakeList.txt
        │       ├── Llama.h
        │       ├── LlamaContextDecoder.h
        │       ├── LlamaDecoder.h
        │       ├── LlamaDecoderLayerWeight.h
        │       └── LlamaWeight.h
        └── triton_backend
            └── llama
                ├── CMakeList.txt
                ├── LlamaTritonModel.h
                └── LlamaTritonModelInstance.h

Faster Transformer Backend
├── all_models
│   └── llama
│       ├── ensemble
│       ├── fastertransformer
│       ├── postprocessing
│       └── preprocessing
└── src
    └── libfastertransformer.cc
```

## 2.1 build your faster transformer library

The [faster transformer repository](https://github.com/void-main/FasterTransformer) work as a  **library** to support different models.

### 2.1.1 **(essential)** faster transformer library for your_model

`examples/cpp/your_model` is essential if you want to run your model on faster transformer.

### 2.1.2 **(essential)** examples for your_model

`src/fastertransformer/models/your_model` is essential because it stores `your_model_config.ini`, and other files (`bad_words.csv`) to ensure your model to work well.

### 2.1.3 **(optional)** triton backend

`src/triton_backend/your/model` is optional. 

Only when you want to deploy your model on triton server with faster transformer backend, you need to implement this part.

# 3. Quick Start

We have deployed llama-7b to triton inference server, see the [llama_guide](./docs/llama_guide.md) to boost your deploying work and get familiar with [NVIDIA Triton Inference Server](https://github.com/triton-inference-server)

