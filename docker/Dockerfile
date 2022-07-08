# Copyright (c) 2021-2202, NVIDIA CORPORATION. All rights reserved.
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


# -------------------------------------------------- #
# This is a Docker image dedicated to develop
# FasterTransformer backend. If you don't want to build the
# backend together with tritonserver, start from here
# -------------------------------------------------- #

ARG TRITON_VERSION=22.01
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:${TRITON_VERSION}-py3
FROM ${BASE_IMAGE} as server-builder

RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    zip unzip wget build-essential autoconf autogen gdb \
    python3.8 python3-pip python3-dev rapidjson-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/build/

# CMake
RUN CMAKE_VERSION=3.18 && \
    CMAKE_BUILD=3.18.4 && \
    wget -nv https://cmake.org/files/v${CMAKE_VERSION}/cmake-${CMAKE_BUILD}.tar.gz && \
    tar -xf cmake-${CMAKE_BUILD}.tar.gz && \
    cd cmake-${CMAKE_BUILD} && \
    ./bootstrap --parallel=$(grep -c ^processor /proc/cpuinfo) -- -DCMAKE_USE_OPENSSL=OFF && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install && \
    cd /workspace/build/ && \
    rm -rf /workspace/build/cmake-${CMAKE_BUILD}

# backend build
WORKDIR /workspace/build/fastertransformer_backend
ADD cmake cmake
ADD src src
ADD CMakeLists.txt CMakeLists.txt
ADD README.md README.md
ADD LICENSE LICENSE

ARG FORCE_BACKEND_REBUILD=0
RUN mkdir build -p && \
    cd build && \
    cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install

FROM ${BASE_IMAGE} as server

ENV NCCL_LAUNCH_MODE=GROUP

COPY --from=server-builder /opt/tritonserver/backends/fastertransformer /opt/tritonserver/backends/fastertransformer

# set workspace
ENV WORKSPACE /workspace
WORKDIR /workspace

RUN   apt-key del 7fa2af80
ADD   https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN   dpkg -i cuda-keyring_1.0-1_all.deb

# Install pytorch
RUN pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html  && \
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex fire ipywidgets tritonclient[all]

# for debug
RUN apt-get update -q && \
    apt-get install -y --no-install-recommends openssh-server zsh tmux mosh locales-all clangd sudo cmake xz-utils zstd libz-dev git-lfs&& \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
RUN mkdir /var/run/sshd -p

# temporal changes only for the blogpost about GPT-J and T5
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

ADD . /workspace/fastertransformer_backend

