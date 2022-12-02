#!/usr/bin/python

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

'''
This example is based on the model https://huggingface.co/rinna/japanese-gpt-neox-small
'''

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

from tritonclient.utils import np_to_triton_dtype

FLAGS = None

OUTPUT_LEN = 500

# start_id = 2 # default setting of tokenizer
# end_id = 3 # default setting of tokenizer

def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)

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
                        help='return the cumulative log probs and output log probs or not')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    # Run async requests to make sure backend handles request batches
    # correctly. We use just HTTP for this since we are not testing the
    # protocol anyway.

    ######################
    model_name = "preprocessing"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        input0 = [
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ]
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
        bad_words_list = np.array([
            [""],
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
            [""]], dtype=object)
        inputs = [
            prepare_tensor("QUERY", input0_data, FLAGS.protocol),
            prepare_tensor("BAD_WORDS_DICT", bad_words_list, FLAGS.protocol),
            prepare_tensor("STOP_WORDS_DICT", stop_words_list, FLAGS.protocol),
            prepare_tensor("REQUEST_OUTPUT_LEN", output0_len, FLAGS.protocol),
        ]

        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("INPUT_ID")
            output1 = result.as_numpy("REQUEST_INPUT_LEN")
            output2 = result.as_numpy("REQUEST_OUTPUT_LEN")
            output3 = result.as_numpy("BAD_WORDS_IDS")
            output4 = result.as_numpy("STOP_WORDS_IDS")
            print("============After preprocessing============")
            print(output0, output1, output2)
            print("===========================================\n\n\n")
        except Exception as e:
            print(e)

    ######################
    model_name = "fastertransformer"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        runtime_top_k = (FLAGS.topk * np.ones([output0.shape[0], 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([output0.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([output0.shape[0], 1]).astype(np.float32)
        random_seed = np.array(range(0, output0.shape[0])).astype(np.uint64).reshape([-1, 1])
        is_return_log_probs = FLAGS.return_log_probs * np.ones([output0.shape[0], 1]).astype(np.bool)
        beam_width = (FLAGS.beam_width * np.ones([output0.shape[0], 1])).astype(np.uint32)
        # start_ids = start_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
        # end_ids = end_id * np.ones([output0.shape[0], 1]).astype(np.uint32)
        prompt_learning_task_name_ids = np.zeros([output0.shape[0], 1]).astype(np.uint32)
        inputs = [
            prepare_tensor("input_ids", output0, FLAGS.protocol),
            prepare_tensor("input_lengths", output1, FLAGS.protocol),
            prepare_tensor("request_output_len", output2, FLAGS.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
            prepare_tensor("temperature", temperature, FLAGS.protocol),
            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
            prepare_tensor("random_seed", random_seed, FLAGS.protocol),
            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
            # prepare_tensor("start_id", start_ids, FLAGS.protocol),
            # prepare_tensor("end_id", end_ids, FLAGS.protocol),
            prepare_tensor("bad_words_list", output3, FLAGS.protocol),
            prepare_tensor("stop_words_list", output4, FLAGS.protocol),
            prepare_tensor("prompt_learning_task_name_ids", prompt_learning_task_name_ids, FLAGS.protocol),
        ]
        try:
            result = client.infer(model_name, inputs)
            output0 = result.as_numpy("output_ids")
            output1 = result.as_numpy("sequence_length").astype(np.uint32)
            print("============After fastertransformer============")
            print(f"output_ids: {output0}")
            print(f"sequence_length: {output1}")
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
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        inputs = [
            prepare_tensor("TOKENS_BATCH", output0, FLAGS.protocol),
            prepare_tensor("sequence_length", output1, FLAGS.protocol),
        ]
        inputs[0].set_data_from_numpy(output0)

        try:
            result = client.infer(model_name, inputs)
            output_sentences = result.as_numpy("OUTPUT")
            print("============After postprocessing============")
            for i, output_sentence in enumerate(output_sentences):
                print(f"sentence {i}: \n{output_sentence.decode('utf8')} \n")
            print("===========================================\n\n\n")
        except Exception as e:
            print(e)

    ######################
    model_name = "ensemble"
    with create_inference_server_client(FLAGS.protocol,
                                        FLAGS.url,
                                        concurrency=1,
                                        verbose=FLAGS.verbose) as client:
        input0 = [
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ["きっとそれは絶対間違ってないね。 わた"],
                ]
        bad_words_list = np.array([
            [""],
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
            [""]], dtype=object)
        input0_data = np.array(input0).astype(object)
        output0_len = np.ones_like(input0).astype(np.uint32) * OUTPUT_LEN
        runtime_top_k = (FLAGS.topk * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        runtime_top_p = FLAGS.topp * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([input0_data.shape[0], 1]).astype(np.float32)
        random_seed = np.array(range(0, output0.shape[0])).astype(np.uint64).reshape([-1, 1])
        is_return_log_probs = True * np.ones([input0_data.shape[0], 1]).astype(bool)
        beam_width = (FLAGS.beam_width * np.ones([input0_data.shape[0], 1])).astype(np.uint32)
        # start_ids = start_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        # end_ids = end_id * np.ones([input0_data.shape[0], 1]).astype(np.uint32)
        prompt_learning_task_name_ids = np.zeros([output0.shape[0], 1]).astype(np.uint32)
        inputs = [
            prepare_tensor("INPUT_0", input0_data, FLAGS.protocol),
            prepare_tensor("INPUT_1", output0_len, FLAGS.protocol),
            prepare_tensor("INPUT_2", bad_words_list, FLAGS.protocol),
            prepare_tensor("INPUT_3", stop_words_list, FLAGS.protocol),
            prepare_tensor("runtime_top_k", runtime_top_k, FLAGS.protocol),
            prepare_tensor("runtime_top_p", runtime_top_p, FLAGS.protocol),
            prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, FLAGS.protocol),
            prepare_tensor("temperature", temperature, FLAGS.protocol),
            prepare_tensor("len_penalty", len_penalty, FLAGS.protocol),
            prepare_tensor("repetition_penalty", repetition_penalty, FLAGS.protocol),
            prepare_tensor("random_seed", random_seed, FLAGS.protocol),
            prepare_tensor("is_return_log_probs", is_return_log_probs, FLAGS.protocol),
            prepare_tensor("beam_width", beam_width, FLAGS.protocol),
            # prepare_tensor("start_id", start_ids, FLAGS.protocol),
            # prepare_tensor("end_id", end_ids, FLAGS.protocol),
            prepare_tensor("prompt_learning_task_name_ids", prompt_learning_task_name_ids, FLAGS.protocol),
        ]
        
        try:
            result = client.infer(model_name, inputs)
            output_sentences = result.as_numpy("OUTPUT_0")
            print("============After ensemble============")
            for i, output_sentence in enumerate(output_sentences):
                print(f"sentence {i}: \n{output_sentence.decode('utf8')} \n")
            if FLAGS.return_log_probs:
                print(result.as_numpy("cum_log_probs"))
                print(result.as_numpy("output_log_probs"))
        except Exception as e:
            print(e)
