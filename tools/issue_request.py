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
import google.protobuf.json_format
import json
import multiprocessing as mp
import numpy as np
import time
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from argparse import ArgumentParser
from collections.abc import Mapping
from functools import partial
from tritonclient.grpc.service_pb2 import ModelInferResponse
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
    parser.add_argument("request_file", nargs="?", default=None, metavar="request-file")
    parser.add_argument("--params")
    args = parser.parse_args()

    return args


def generate_parameters(args):
    DEFAULT_CONFIG = {
        'protocol': 'http',
        'url': None,
        'model_name': 'fastertransformer',
        'verbose': False,
        'stream_api': False,
    }
    params = {'config': DEFAULT_CONFIG, 'request': []}

    if args.request_file is not None:
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

    if params['config']['url'] is None:
        if params['config']['protocol'] == 'grpc':
            params['config']['url'] = 'localhost:8001'
        else:
            params['config']['url'] = 'localhost:8000'

    return params['config'], params['request']


def prepare_tensor(client, name, input):
    t = client.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def stream_consumer(queue):
    start_time = None
    while True:
        result = queue.get()
        if result is None:
            break

        if isinstance(result, float):
            start_time = result
            continue

        message = ModelInferResponse()
        google.protobuf.json_format.Parse(json.dumps(result), message)
        result = grpcclient.InferResult(message)

        idx = result.as_numpy("sequence_length")[0, 0]
        tokens = result.as_numpy("output_ids")[0, 0, :idx]
        prob = result.as_numpy("cum_log_probs")[0, 0]
        print("[After {:.2f}s] Partial result (probability {:.2e}):\n{}\n".format(
            time.perf_counter() - start_time, np.exp(prob), tokens))


def stream_callback(queue, result, error):
    if error:
        queue.put(error)
    else:
        queue.put(result.get_response(as_json=True))


def main_stream(config, request):
    client_type = grpcclient

    kwargs = {"verbose": config["verbose"]}
    result_queue = mp.Queue()

    consumer = mp.Process(target=stream_consumer, args=(result_queue,))
    consumer.start()

    with grpcclient.InferenceServerClient(config['url'], verbose=config["verbose"]) as cl:
        payload = [prepare_tensor(grpcclient, field['name'], field['data'])
            for field in request]

        cl.start_stream(callback=partial(stream_callback, result_queue))
        result_queue.put(time.perf_counter())
        cl.async_stream_infer(config['model_name'], payload)
    result_queue.put(None)
    consumer.join()


def main_sync(config, request):
    is_http = config['protocol'] == 'http'
    client_type = httpclient if is_http else grpcclient

    kwargs = {"verbose": config["verbose"]}
    if is_http:
        kwargs["concurrency"] = 10
    with client_type.InferenceServerClient(config['url'], **kwargs) as cl:
        payload = [prepare_tensor(client_type, field['name'], field['data'])
            for field in request]

        result = cl.infer(config['model_name'], payload)

    if is_http:
        for output in result.get_response()['outputs']:
            print("{}:\n{}\n".format(output['name'], result.as_numpy(output['name'])))
    else:
        for output in result.get_response().outputs:
            print("{}:\n{}\n".format(output.name, result.as_numpy(output.name)))


if __name__ == "__main__":
    args = parse_args()
    config, request = generate_parameters(args)
    if not config['stream_api']:
        main_sync(config, request)
    else:
        main_stream(config, request)
