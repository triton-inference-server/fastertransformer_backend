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

import argparse
import numpy as np
import sys
from builtins import range
import statistics as s
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import random
import torch
from transformers import BertModel

WEIGHT2NPDTYPE = {
    "fp32": np.float32,
    "fp16": np.float16,
}

WEIGHT2THDTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
}

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

def sequence_mask(lengths, max_len=None, is_2d=True):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    mask = (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))
    if is_2d:
        return mask
    else:
        mask = mask.view(-1, 1, 1, max_len)
        m2 = mask.transpose(2, 3)
        return mask * m2

def send_requests(url, input_hidden_state, sequence_lengths,
                  verbose, flags, request_parallelism=10):
    model_name = "fastertransformer"
    with create_inference_server_client(flags.protocol,
                                        url,
                                        concurrency=request_parallelism,
                                        verbose=verbose) as client:
        requests = []
        results = []
        
        for i in range(request_parallelism):
            inputs = [
                prepare_tensor("input_hidden_state", input_hidden_state, flags.protocol),
                prepare_tensor("sequence_lengths", sequence_lengths, flags.protocol),
            ]

            print("set request")
            result = client.infer(model_name, inputs)
            results.append(result)

        for i in range(request_parallelism):

            output_hidden_state = results[i].as_numpy("output_hidden_state")
            print("get results as output_hidden_state\n")
            if output_hidden_state is None:
                print("error: expected 'output_hidden_state'")
                sys.exit(1)
            else:
                pass
    return output_hidden_state


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
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('-w',
                        '--warm_up',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable warm_up before benchmark')
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=8,
                        required=False,
                        help='Specify batch size')
    parser.add_argument('-s',
                        '--seq_len',
                        type=int,
                        default=32,
                        required=False,
                        help='Specify input length')
    parser.add_argument('-hidden_dim',
                        '--hidden_dim',
                        type=int,
                        default=768,
                        required=False,
                        help='Specify hidden dimension')
    parser.add_argument('-n',
                        '--num_runs',
                        type=int,
                        default=1,
                        required=False,
                        help="Specify number of runs to get the average latency"
                        )
    parser.add_argument("--hf_ckpt_path",
                        type=str,
                        default="bert-base-uncased",
                        help="The checkpoint of huggingface bert model")
    parser.add_argument("--inference_data_type",
                        type=str,
                        default="fp32",
                        choices=["fp32", "fp16"],
                        help="The data type for inference")
    parser.add_argument("--max_diff_threshold",
                        type=float,
                        help="Threshold of max differences")

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    input_hidden_state = np.random.normal(0.0, 0.02, [FLAGS.batch_size, FLAGS.seq_len, FLAGS.hidden_dim])
    input_hidden_state = input_hidden_state.astype(WEIGHT2NPDTYPE[FLAGS.inference_data_type])
    sequence_lengths = np.random.randint(1, FLAGS.seq_len, [FLAGS.batch_size, 1], np.int32)

    mask = sequence_mask(torch.Tensor(sequence_lengths.reshape([-1])).to(torch.int), FLAGS.seq_len, False)
    mask = mask.to(WEIGHT2THDTYPE[FLAGS.inference_data_type]).cuda()
    output_mask = sequence_mask(torch.Tensor(sequence_lengths.reshape([-1])).to(torch.int), FLAGS.seq_len)
    output_mask = output_mask.to(mask.dtype).unsqueeze(-1).cuda()
    
    # verify correctness
    hf_model = BertModel.from_pretrained(FLAGS.hf_ckpt_path)
    hf_model.cuda().eval().to(WEIGHT2THDTYPE[FLAGS.inference_data_type])
    hf_input = torch.Tensor(input_hidden_state).cuda().to(WEIGHT2THDTYPE[FLAGS.inference_data_type])
    extended_attention_mask = (1.0 - mask) * -10000.0
    hf_output = hf_model.encoder(hf_input, extended_attention_mask, return_dict=False)[0] * output_mask

    ft_output_hidden_state = send_requests(FLAGS.url, input_hidden_state, sequence_lengths,
                                           FLAGS.verbose, FLAGS, request_parallelism=1)
    diff = (torch.Tensor(ft_output_hidden_state).cuda() * output_mask) - hf_output
    
    # warm up
    if FLAGS.warm_up:
        print("[INFO] sending requests to warm up")
        send_requests(FLAGS.url, input_hidden_state, sequence_lengths,
                    FLAGS.verbose, FLAGS, request_parallelism=2)
    import time
    time.sleep(5)  # TODO: Not sure if this is necessary
    from datetime import datetime
    request_parallelism = 10
    latencies = []
    for i in range(FLAGS.num_runs):
        start_time = datetime.now()
        send_requests(FLAGS.url, input_hidden_state, sequence_lengths,
                    FLAGS.verbose, FLAGS, request_parallelism=2)
        stop_time = datetime.now()
        latencies.append((stop_time - start_time).total_seconds()
                         * 1000.0 / request_parallelism)

    print(f"abs max diff: {diff.abs().max()}")
    print(f"abs mean diff: {diff.abs().mean()}")
    if FLAGS.num_runs > 1:
        print(f"[INFO] execution time: {s.mean(latencies)} ms")
    else:
        print(f"[INFO] execution time: {latencies[0]} ms")
    if FLAGS.max_diff_threshold != None:
        assert FLAGS.max_diff_threshold >= diff.abs().max()
