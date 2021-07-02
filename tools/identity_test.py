#!/usr/bin/python

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

import argparse
import numpy as np
import os
import re
import sys
import requests as httpreq
from builtins import range
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

FLAGS = None

START_LEN = 8
OUTPUT_LEN = 24
BATCH_SIZE = 8

start_id = 220
end_id = 50256
# random_start_ids = np.random.randint(0, 50255, size=(BATCH_SIZE, START_LEN), dtype=np.uint32)
random_start_ids = np.array([[9915, 27221, 59, 77, 383, 1853, 3327, 1462],
                             [6601, 4237, 345, 460, 779, 284, 787, 257],
                             [59, 77, 611, 7, 9248, 796, 657, 8],
                             [38, 10128, 6032, 651, 8699, 4, 4048, 20753],
                             [21448, 7006, 930, 12901, 930, 7406, 7006, 198],
                             [13256, 11, 281, 1605, 3370, 11, 1444, 6771],
                             [9915, 27221, 59, 77, 383, 1853, 3327, 1462],
                             [6601, 4237, 345, 460, 779, 284, 787, 257]], np.uint32)
input_len = np.array([ [sentence.size] for sentence in random_start_ids ], np.uint32)
output_len = np.ones_like(input_len).astype(np.uint32) * OUTPUT_LEN

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        help='Inference server URL.')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Run async requests to make sure backend handles request batches
    # correctly. We use just HTTP for this since we are not testing the
    # protocol anyway.

    # warmup
    if FLAGS.protocol == "http":
        request_parallelism = 10
        model_name = "transformer"
        # shape = [8, 8]
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=request_parallelism,
                                               verbose=FLAGS.verbose) as client:
            requests = []
            results = []
            for i in range(request_parallelism):
                input_data = random_start_ids
                inputs = [
                    client_util.InferInput("INPUT_ID", input_data.shape,
                                           np_to_triton_dtype(input_data.dtype)),
                    client_util.InferInput("REQUEST_INPUT_LEN", input_len.shape,
                                           np_to_triton_dtype(input_len.dtype)),
                    client_util.InferInput("REQUEST_OUTPUT_LEN", output_len.shape,
                                           np_to_triton_dtype(output_len.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                inputs[1].set_data_from_numpy(input_len)
                inputs[2].set_data_from_numpy(output_len)
                result = client.infer(model_name, inputs)
                results.append(result)

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                output_data = results[i].as_numpy("OUTPUT0")

    from datetime import datetime
    request_parallelism = 10
    start_time = datetime.now()
    if FLAGS.protocol == "http":
        model_name = "transformer"
        # shape = [8, 8]
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=request_parallelism,
                                               verbose=FLAGS.verbose) as client:
            requests = []
            results = []
            for i in range(request_parallelism):
                input_data = random_start_ids
                inputs = [
                    client_util.InferInput("INPUT_ID", input_data.shape,
                                           np_to_triton_dtype(input_data.dtype)),
                    client_util.InferInput("REQUEST_INPUT_LEN", input_len.shape,
                                           np_to_triton_dtype(input_len.dtype)),
                    client_util.InferInput("REQUEST_OUTPUT_LEN", output_len.shape,
                                           np_to_triton_dtype(output_len.dtype))
                ]
                inputs[0].set_data_from_numpy(input_data)
                inputs[1].set_data_from_numpy(input_len)
                inputs[2].set_data_from_numpy(output_len)
                #requests.append(client.async_infer(model_name, inputs))
                print("set request")
                result = client.infer(model_name, inputs)
                print("get request")
                results.append(result)

            for i in range(request_parallelism):
                # Get the result from the initiated asynchronous inference request.
                # Note the call will block till the server responds.
                print("wait result return 0000\n")
                ##results = requests[i].get_result()
                print("wait result return 1111\n")
                # print(results[i])
                print("get results\n")

                output_data = results[i].as_numpy("OUTPUT0")
                output_data = output_data.reshape([-1, BATCH_SIZE])
                np.savetxt("triton_out", output_data, fmt='%u')
                output_data = output_data.T
                print("get results as OUTPUT0\n")
                if output_data is None:
                    print("error: expected 'OUTPUT0'")
                    sys.exit(1)
                else:
                    print("OUTPUT0 is received")
                    print(output_data.shape)
                    print(output_data)
    stop_time = datetime.now()
    print("[INFO] execution time: {} ms".format((stop_time - start_time).total_seconds() * 1000.0 / request_parallelism))
