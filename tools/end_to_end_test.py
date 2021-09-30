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
from builtins import range
import grpc
from tritonclient.grpc import service_pb2
from tritonclient.grpc import service_pb2_grpc
import tritongrpcclient as grpcclient
import tritonhttpclient as httpclient
from tritonclientutils import np_to_triton_dtype

FLAGS = None

START_LEN = 8
OUTPUT_LEN = 24
BATCH_SIZE = 8

start_id = 220
end_id = 50256

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
    # parser.add_argument('-beam',
    #                     '--beam_width',
    #                     type=int,
    #                     default=1,
    #                     help='beam width.')
    
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

    if FLAGS.protocol == "http":
        ######################
        model_name = "preprocessing"
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=1,
                                               verbose=FLAGS.verbose) as client:
            input0 = [
                    ["Blackhawks\n The 2015 Hilltoppers"],
                    ["Data sources you can use to make a decision:"],
                    ["\n if(angle = 0) { if(angle"],
                    ["GMs typically get 78% female enrollment, but the "],
                    ["Previous Chapter | Index | Next Chapter"],
                    ["Michael, an American Jew, called Jews"],
                    ["Blackhawks\n The 2015 Hilltoppers"],
                    ["Data sources you can use to make a comparison:"]
                    ]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
            inputs = [
                client_util.InferInput("QUERY", input0_data.shape,
                                       np_to_triton_dtype(input0_data.dtype)),
                client_util.InferInput("REQUEST_OUTPUT_LEN", output0_len.shape,
                                       np_to_triton_dtype(output0_len.dtype))
            ]
            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(output0_len)

            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("INPUT_ID")
                output1 = result.as_numpy("REQUEST_INPUT_LEN")
                output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
                # output0 = output0.reshape([output0.shape[0], 1, output0.shape[1]]) # Add dim for beam width
                print("============After preprocessing============")
                print(output0, output1, output2)
                print("===========================================\n\n\n")
            except Exception as e:
                print(e)

        ######################
        model_name = "fastertransformer"
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=1,
                                               verbose=FLAGS.verbose) as client:
            inputs = [
                client_util.InferInput("INPUT_ID", output0.shape,
                                       np_to_triton_dtype(output0.dtype)),
                client_util.InferInput("REQUEST_INPUT_LEN", output1.shape,
                                       np_to_triton_dtype(output1.dtype)),
                client_util.InferInput("REQUEST_OUTPUT_LEN", output2.shape,
                                       np_to_triton_dtype(output2.dtype))
            ]
            inputs[0].set_data_from_numpy(output0)
            inputs[1].set_data_from_numpy(output1)
            inputs[2].set_data_from_numpy(output2)
            
            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT0")
                print("============After fastertransformer============")
                print(output0)
                print("===========================================\n\n\n")
            except Exception as e:
                print(e)

        ######################
        model_name = "postprocessing"
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=1,
                                               verbose=FLAGS.verbose) as client:
            inputs = [
                client_util.InferInput("TOKENS_BATCH", output0.shape,
                                       np_to_triton_dtype(output0.dtype))
            ]
            inputs[0].set_data_from_numpy(output0)

            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT")
                print("============After postprocessing============")
                print(output0)
                print("===========================================\n\n\n")
            except Exception as e:
                print(e)

        ######################
        model_name = "ensemble"
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=1,
                                               verbose=FLAGS.verbose) as client:
            input0 = [
                    ["Blackhawks\n The 2015 Hilltoppers"],
                    ["Data sources you can use to make a decision:"],
                    ["\n if(angle = 0) { if(angle"],
                    ["GMs typically get 78% female enrollment, but the "],
                    ["Previous Chapter | Index | Next Chapter"],
                    ["Michael, an American Jew, called Jews"],
                    ["Blackhawks\n The 2015 Hilltoppers"],
                    ["Data sources you can use to make a comparison:"]
                    ]
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
            inputs = [
                client_util.InferInput("INPUT_0", input0_data.shape,
                                       np_to_triton_dtype(input0_data.dtype)),
                client_util.InferInput("INPUT_1", output0_len.shape,
                                       np_to_triton_dtype(output0_len.dtype))
            ]
            inputs[0].set_data_from_numpy(input0_data)
            inputs[1].set_data_from_numpy(output0_len)

            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT_0")
                print("============After ensemble============")
                print(output0)
            except Exception as e:
                print(e)
