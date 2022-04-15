#!/usr/bin/env python3

# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import json
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from argparse import ArgumentParser
from collections.abc import Mapping
from tritonclient.utils import np_to_triton_dtype


def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("request_file", metavar="request-file")
    parser.add_argument("--params")
    args = parser.parse_args()

    return args


def generate_parameters(args):
    DEFAULT_CONFIG = {
        'protocol': 'http',
        'url': 'localhost:8000',
        'model_name': 'fastertransformer',
        'verbose': False,
    }
    params = {'config': DEFAULT_CONFIG}

    with open(args.request_file) as f:
        file_params = json.load(f)
    deep_update(params, file_params)

    args_params = json.loads(args.params) if args.params else {}
    deep_update(params, args_params)

    for index, value in enumerate(params['request']):
        params['request'][index] = {
            'name': value['name'],
            'data': np.array(value['data'], dtype=value['dtype']),
        }

    return params['config'], params['request']


def prepare_tensor(client, name, input):
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def main(config, request):
    client_type = httpclient if config['protocol'] == 'http' else grpcclient
    with client_type.InferenceServerClient(config['url'], verbose=config['verbose'], concurrency=10) as cl:
        payload = [prepare_tensor(client_type, field['name'], field['data'])
            for field in request]

        result = cl.infer(config['model_name'], payload)

    for output in result.get_response()['outputs']:
        print("{}:\n{}\n".format(output['name'], result.as_numpy(output['name'])))


if __name__ == "__main__":
    args = parse_args()
    config, request = generate_parameters(args)
    main(config, request)
