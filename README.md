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

All implements are actually in [FasterTransformer]() repository.

The `CMakeList.txt` will fetch relative repository to organize and compile the project with:

+ this repository itself
+ Faster Transformer repository
+ 3rdparty
+ cutlass

etc...

# 2. LLaMa support
To check how faster transformer support LLaMa, and how triton ft support LLaMa, here are three links you may be interested:

## 2.1 LLaMa example
https://github.com/SamuraiBUPT/adapter_ft_library/tree/main/examples/cpp/llama

the example includes works from `2.2 LLaMa support implement`, the actual work is in `2.2`.

## 2.2 LLaMa support implement
https://github.com/SamuraiBUPT/adapter_ft_library/tree/main/src/fastertransformer/models/llama

This is the core implement to support LLaMa in faster transformer.

## 2.3 LLaMa triton faster transformer backend support
https://github.com/SamuraiBUPT/adapter_ft_library/tree/main/src/fastertransformer/triton_backend/llama

It's surprising that triton support was put in Faster Transformer repository. 

Maybe that is because they will be finally compiled as a whole, so never mind the job of mixed developed.