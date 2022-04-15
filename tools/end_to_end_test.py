#!/usr/bin/python

# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype

FLAGS = None

START_LEN = 8
OUTPUT_LEN = 24
BATCH_SIZE = 8

start_id = 220
end_id = 50256

def prepare_tensor(name, input):
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

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
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        help='beam width.')
    parser.add_argument('-topk',
                        '--topk',
                        type=int,
                        default=1,
                        required=False,
                        help='topk for sampling')
    parser.add_argument('-topp',
                        '--topp',
                        type=float,
                        default=0.0,
                        required=False,
                        help='topp for sampling')
    parser.add_argument(
        '-i',
        '--protocol',
        type=str,
        required=False,
        default='http',
        help='Protocol ("http"/"grpc") used to ' +
        'communicate with inference service. Default is "http".')
    parser.add_argument('--return_log_probs',
                        action="store_true",
                        default=False,
                        required=False,
                        help='return the cumulative log porbs and output log probs or not')

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
            bad_words_list = np.array([
                ["Hawks, Hawks"],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""]], dtype=object)
            stop_words_list = np.array([
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                ["month, month"]], dtype=object)
            inputs = [
                prepare_tensor("QUERY", input0_data),
                prepare_tensor("BAD_WORDS_DICT", bad_words_list),
                prepare_tensor("STOP_WORDS_DICT", stop_words_list),
                prepare_tensor("REQUEST_OUTPUT_LEN", output0_len),
            ]

            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("INPUT_ID")
                output1 = result.as_numpy("REQUEST_INPUT_LEN")
                output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
                output3 = result.as_numpy("BAD_WORDS_IDS")
                output4 = result.as_numpy("STOP_WORDS_IDS")
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
            runtime_top_k = (FLAGS.topk * np.ones([output0.shape[0], 1])).astype(np.uint32)
            runtime_top_p = FLAGS.topp * np.ones([output0.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
            random_seed = 0 * np.ones([output0.shape[0], 1]).astype(np.int32)
            is_return_log_probs = FLAGS.return_log_probs * np.ones([output0.shape[0], 1]).astype(np.bool)
            beam_width = (FLAGS.beam_width * np.ones([output0.shape[0], 1])).astype(np.uint32)
            start_ids = start_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
            end_ids = end_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
            inputs = [
                prepare_tensor("input_ids", output0),
                prepare_tensor("input_lengths", output1),
                prepare_tensor("request_output_len", output2),
                prepare_tensor("runtime_top_k", runtime_top_k),
                prepare_tensor("runtime_top_p", runtime_top_p),
                prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
                prepare_tensor("temperature", temperature),
                prepare_tensor("len_penalty", len_penalty),
                prepare_tensor("repetition_penalty", repetition_penalty),
                prepare_tensor("random_seed", random_seed),
                prepare_tensor("is_return_log_probs", is_return_log_probs),
                prepare_tensor("beam_width", beam_width),
                prepare_tensor("start_id", start_ids),
                prepare_tensor("end_id", end_ids),
                prepare_tensor("bad_words_list", output3),
                prepare_tensor("stop_words_list", output4),
            ]

            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("output_ids")
                output1 = result.as_numpy("sequence_length")
                print("============After fastertransformer============")
                print(output0)
                print(output1)
                if FLAGS.return_log_probs:
                    output2 = result.as_numpy("cum_log_probs")
                    output3 = result.as_numpy("output_log_probs")
                    print(output2)
                    print(output3)
                print("===========================================\n\n\n")
            except Exception as e:
                print(e)

        ######################
        model_name = "postprocessing"
        with client_util.InferenceServerClient(FLAGS.url,
                                               concurrency=1,
                                               verbose=FLAGS.verbose) as client:
            inputs = [
                prepare_tensor("TOKENS_BATCH", output0),
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
            bad_words_list = np.array([
                ["Hawks, Hawks"],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""]], dtype=object)
            stop_words_list = np.array([
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                [""],
                ["month, month"]], dtype=object)
            input0_data = np.array(input0).astype(object)
            output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
            runtime_top_k = (FLAGS.topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
            runtime_top_p = FLAGS.topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
            random_seed = 0 * np.ones([input0_data.shape[0], 1]).astype(np.int32)
            is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
            beam_width = (FLAGS.beam_width * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
            start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
            end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
            inputs = [
                prepare_tensor("INPUT_0", input0_data),
                prepare_tensor("INPUT_1", output0_len),
                prepare_tensor("INPUT_2", bad_words_list),
                prepare_tensor("INPUT_3", stop_words_list),
                prepare_tensor("runtime_top_k", runtime_top_k),
                prepare_tensor("runtime_top_p", runtime_top_p),
                prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate),
                prepare_tensor("temperature", temperature),
                prepare_tensor("len_penalty", len_penalty),
                prepare_tensor("repetition_penalty", repetition_penalty),
                prepare_tensor("random_seed", random_seed),
                prepare_tensor("is_return_log_probs", is_return_log_probs),
                prepare_tensor("beam_width", beam_width),
                prepare_tensor("start_id", start_ids),
                prepare_tensor("end_id", end_ids),
            ]
            
            try:
                result = client.infer(model_name, inputs)
                output0 = result.as_numpy("OUTPUT_0")
                print("============After ensemble============")
                print(output0)
                print(result.as_numpy("sequence_length"))
                if FLAGS.return_log_probs:
                    print(result.as_numpy("cum_log_probs"))
                    print(result.as_numpy("output_log_probs"))
            except Exception as e:
                print(e)
