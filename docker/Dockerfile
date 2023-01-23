# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG TRITON_VERSION=22.12
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE}

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        autoconf \
        autogen \
        clangd \
        gdb \
        git-lfs \
        libb64-dev \
        libz-dev \
        locales-all \
        mosh \
        openssh-server \
        python3-dev \
        rapidjson-dev \
        sudo \
        tmux \
        unzip \
        zstd \
        zip \
        zsh
RUN pip3 install torch==1.12.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire tritonclient[all] && \
    pip3 install transformers huggingface_hub tokenizers SentencePiece sacrebleu datasets tqdm omegaconf rouge_score && \
    pip3 install cmake==3.24.3

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# backend build
ADD . /workspace/build/fastertransformer_backend
RUN mkdir -p /workspace/build/fastertransformer_backend/build

WORKDIR /workspace/build/fastertransformer_backend/build
ARG FORCE_BACKEND_REBUILD=0
RUN cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D ENABLE_FP8=OFF \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      ..
RUN cd _deps/repo-ft-src/ && \
    git log | head -n 3 2>&1 | tee /workspace/build/fastertransformer_backend/FT_version.txt && \
    cd /workspace/build/fastertransformer_backend/build && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install && \
    rm /workspace/build/fastertransformer_backend/build/bin/*_example -rf && \
    rm /workspace/build/fastertransformer_backend/build/lib/lib*Backend.so -rf

ENV NCCL_LAUNCH_MODE=GROUP
ENV WORKSPACE /workspace
WORKDIR /workspace

RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config && \
    mkdir /var/run/sshd -p

