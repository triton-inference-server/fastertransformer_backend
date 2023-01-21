#!/usr/bin/env python3
# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import argparse
import sys
import os
import subprocess

FLAGS = None


#### helper functions
def log(msg, force=False):
    if force or not FLAGS.quiet:
        try:
            print(msg, file=sys.stderr)
        except Exception:
            print('<failed to log>', file=sys.stderr)


def log_verbose(msg):
    if FLAGS.verbose:
        log(msg, force=True)


def fail(msg):
    print('error: {}'.format(msg), file=sys.stderr)
    sys.exit(1)


def fail_if(p, msg):
    if p:
        fail(msg)


def create_dependencies(base_image):
    df = '''
ARG BASE_IMAGE={base_image}
    '''.format(base_image=base_image)
    df += '''
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \\
        autoconf \\
        autogen \\
        clangd \\
        gdb \\
        git-lfs \\
        libb64-dev \\
        libz-dev \\
        locales-all \\
        mosh \\
        openssh-server \\
        python3-dev \\
        rapidjson-dev \\
        sudo \\
        tmux \\
        unzip \\
        zstd \\
        zip \\
        zsh \\
        python3-pip
RUN pip3 install torch==1.12.1+cu116 -f \\
                    https://download.pytorch.org/whl/torch_stable.html && \\
    pip3 install --extra-index-url https://pypi.ngc.nvidia.com regex \\
                    fire tritonclient[all] && \\
    pip3 install transformers huggingface_hub tokenizers SentencePiece \\
                    sacrebleu datasets tqdm omegaconf rouge_score && \\
    pip3 install cmake==3.24.3
RUN apt-get clean && \\
    rm -rf /var/lib/apt/lists/*
'''
    return df


def create_build():
    df = '''
# backend build
ADD . /workspace/build/fastertransformer_backend
RUN mkdir -p /workspace/build/fastertransformer_backend/build
WORKDIR /workspace/build/fastertransformer_backend/build
RUN cmake \\
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \\
      -D CMAKE_BUILD_TYPE=Release \\
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \\
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \\
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \\
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \\
      ..
RUN make -j"$(grep -c ^processor /proc/cpuinfo)" install
    '''
    return df


def create_postbuild(is_multistage_build):
    if is_multistage_build:
        df = '''
FROM ${BASE_IMAGE}
WORKDIR /opt/tritonserver
COPY --from=build_image /opt/tritonserver/backends/fastertransformer/ \\
                        backends/fastertransformer
RUN apt-get update && apt-get install -y --no-install-recommends \\
                        openssh-server && rm -rf /var/lib/apt/lists/*
'''
    else:
        df = '''
ENV WORKSPACE /workspace
WORKDIR /workspace       
'''
    df += '''
ENV NCCL_LAUNCH_MODE=GROUP
RUN sed -i 's/#X11UseLocalhost yes/X11UseLocalhost no/g' /etc/ssh/sshd_config \\
    && mkdir /var/run/sshd -p
    '''
    return df


def build_docker_image(ddir, dockerfile_name, container_name):
    # Create container with docker build
    p = subprocess.Popen(['docker', 'build', '-t', container_name, '-f', \
        os.path.join(ddir, dockerfile_name), '.'])
    p.wait()
    fail_if(p.returncode != 0, 'docker build {} failed'.format(container_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group_qv = parser.add_mutually_exclusive_group()
    group_qv.add_argument('-q',
                          '--quiet',
                          action="store_true",
                          required=False,
                          help='Disable console output.')
    group_qv.add_argument('-v',
                          '--verbose',
                          action="store_true",
                          required=False,
                          help='Enable verbose output.')
    parser.add_argument(
        '--dry-run',
        action="store_true",
        required=False,
        help='Only creates Dockerfile, does not build the Docker image.')
    parser.add_argument('--triton-version',
                        required=False,
                        help='Triton version to use as build base.')
    parser.add_argument('--base-image',
                        required=False,
                        help="Triton base image name.")
    parser.add_argument('--work-dir',
                        required=False,
                        help="Location to generate dockerfile.")
    parser.add_argument('--dockerfile-name',
                        required=False,
                        help="Name of generated dockerfile.")
    parser.add_argument('--image-name',
                        required=False,
                        help="Name of generated image.")
    parser.add_argument(
        '--is-multistage-build',
        action="store_true",
        required=False,
        help=
        'Creates a multistage build that only contains the runtime requirements.'
    )

    FLAGS = parser.parse_args()
    fail_if(
        FLAGS.triton_version is None and FLAGS.base_image is None,
        "Need to specify either full image name or Triton version to use as base container"
    )
    if FLAGS.work_dir is None:
        FLAGS.work_dir = "."
    if FLAGS.dockerfile_name is None:
        FLAGS.dockerfile_name = "Dockerfile.gen"
    if FLAGS.image_name is None:
        FLAGS.image_name = "tritonserver_with_ft"

    if FLAGS.base_image is None:
        base_image = "nvcr.io/nvidia/tritonserver:" + FLAGS.triton_version + "-py3"
    else:
        base_image = FLAGS.base_image

    df = create_dependencies(base_image)
    df += create_build()
    df += create_postbuild(FLAGS.is_multistage_build)
    path = os.path.join(FLAGS.work_dir, FLAGS.dockerfile_name)
    if os.path.exists(path):
        os.remove(path)
    with open(path, "a") as dfile:
        dfile.write(df)
    if (not FLAGS.dry_run):
        build_docker_image(FLAGS.work_dir, FLAGS.dockerfile_name,
                           FLAGS.image_name)
