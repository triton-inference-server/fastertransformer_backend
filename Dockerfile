

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:21.02-py3
ARG SDK_IMAGE=nvcr.io/nvidia/tritonserver:21.02-py3-sdk

FROM ${SDK_IMAGE} AS sdk_image

FROM ${BASE_IMAGE} as ftbe_sdk
#RUN mkdir /usr/local/mpi
#COPY --from=mpi_image /usr/local/mpi/ /usr/local/mpi/

RUN     apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common \
        autoconf                   \
        automake                   \
        build-essential            \
        docker.io                  \
        git                        \
        libre2-dev                 \
        libssl-dev                 \
        libtool                    \
        libboost-dev               \
        libcurl4-openssl-dev       \
        libb64-dev                 \
        patchelf                   \
        python3-dev                \
        python3-pip                \
        python3-setuptools         \
        rapidjson-dev              \
        unzip                      \
        wget                       \
        zlib1g-dev                 \
        pkg-config                 \
        uuid-dev

RUN     pip3 install --upgrade pip && \
        pip3 install --upgrade wheel setuptools docker && \
        pip3 install grpcio-tools grpcio-channelz

RUN     wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | \
        gpg --dearmor - | \
        tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null &&  \
        apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main' && \
        apt-get update && \
        apt-get install -y --no-install-recommends \
        cmake-data=3.18.4-0kitware1ubuntu20.04.1 cmake=3.18.4-0kitware1ubuntu20.04.1


################################################################################
## COPY from Dockerfile.sdk
################################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            software-properties-common \
            autoconf \
            automake \
            build-essential \
            curl \
            git \
            libb64-dev \
            libopencv-dev \
            libopencv-core-dev \
            libssl-dev \
            libtool \
            pkg-config \
            python3 \
            python3-pip \
            python3-dev \
            rapidjson-dev \
            vim \
            wget && \
    pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade grpcio-tools && \
    pip3 install --upgrade pip

# Build expects "python" executable (not python3).
RUN rm -f /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python
# Install the dependencies needed to run the client examples. These
# are not needed for building but including them allows this image to
# be used to run the client examples.
RUN pip3 install --upgrade numpy pillow
##     find install/python/ -maxdepth 1 -type f -name \
##     "tritonclient-*-manylinux1_x86_64.whl" | xargs printf -- '%s[all]' | \
##     xargs pip3 install --upgrade

# Install DCGM
# DCGM version to install for Model Analyzer
## ARG DCGM_VERSION=2.0.13
## RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb && \
##     dpkg -i datacenter-gpu-manager_${DCGM_VERSION}_amd64.deb
COPY --from=sdk_image /workspace/datacenter-gpu-manager_*_amd64.deb /tmp/
RUN dpkg -i /tmp/datacenter-gpu-manager_*_amd64.deb && rm -rf /tmp/datacenter-gpu-manager_*_amd64.deb

# Install Model Analyzer
ARG TRITON_MODEL_ANALYZER_REPO_TAG=r20.12
ARG TRITON_MODEL_ANALYZER_REPO="https://github.com/triton-inference-server/model_analyzer@${TRITON_MODEL_ANALYZER_REPO_TAG}"
RUN pip3 install "git+${TRITON_MODEL_ANALYZER_REPO}"
################################################################################
## COPY from Dockerfile.QA
################################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpng-dev \
        curl \
        libopencv-dev \
        libopencv-core-dev \
        libzmq3-dev \
        python3-dev \
        python3-pip \
        python3-protobuf \
        python3-setuptools \
        swig \
        nginx \
        protobuf-compiler \
        valgrind

RUN wget https://go.dev/dl/go1.16.4.linux-amd64.tar.gz && \
	rm -rf /usr/local/go && \
	tar -C /usr/local -xzf go1.16.4.linux-amd64.tar.gz

ENV PATH=/usr/local/go/bin:$PATH

RUN pip3 install --upgrade wheel setuptools && \
    pip3 install --upgrade numpy pillow future grpcio requests gsutil awscli six boofuzz grpcio-channelz azure-cli

# need protoc-gen-go to generate go specific gRPC modules
RUN go get github.com/golang/protobuf/protoc-gen-go && \
        go get google.golang.org/grpc

COPY --from=sdk_image /workspace/install/python/tritonclient-2.7.0-py3-none-manylinux1_x86_64.whl /tmp/
RUN pip3 install --upgrade /tmp/tritonclient-2.7.0-py3-none-manylinux1_x86_64.whl[all]

RUN mkdir /opt/tritonserver/backends/fastertransformer && chmod 777 /opt/tritonserver/backends/fastertransformer

FROM ftbe_sdk as ftbe_work
# for debug
RUN apt update -q && apt install -y --no-install-recommends openssh-server zsh tmux mosh locales-all clangd sudo
RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config
RUN mkdir /var/run/sshd

ENTRYPOINT service ssh restart && bash
