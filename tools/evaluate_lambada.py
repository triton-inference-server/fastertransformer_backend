# -*- coding: utf-8 -*-
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

import os
import json
import csv
import utils.gpt_token_encoder as encoder
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


START_ID = 50256
END_ID = 50256

MAX_TEST_GRAM = 4


def send_requests(url, input_start_ids, input_len, output_len, verbose, request_parallelism=10, model_name="fastertransformer"):
    with client_util.InferenceServerClient(url,
                                           concurrency=request_parallelism,
                                           verbose=verbose) as client:
        input_data = input_start_ids
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

        return result.as_numpy("OUTPUT0")


def load_data(enc, dataset_path, number_of_samples):

    all_ids = []
    raw_text = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw_text.append(json.loads(line))
            all_ids.append(torch.IntTensor(enc.encode(raw_text[-1]['text'])))

            if number_of_samples and len(raw_text) > number_of_samples:
                break

    return all_ids, raw_text


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
    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        default=128,
                        required=False,
                        help='Specify batch size')
    parser.add_argument('-o',
                        '--output_csv',
                        type=str,
                        required=False,
                        default="./lambada_metrics.csv",
                        help="dump metrics into csv file")
    parser.add_argument('-d',
                        '--datasets_dir',
                        type=str,
                        required=False,
                        default="./",
                        help='Folder contains vocab and dataset')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=False,
                        default="fastertransformer",
                        help='model name')
    parser.add_argument('--n-gram-disabled',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Disable n-gram calculation')
    parser.add_argument('--number-of-samples',
                        type=int,
                        required=False,
                        default=None,
                        help='Limits number of samples for test')
    # parser.add_argument('-beam',
    #                     '--beam_width',
    #                     type=int,
    #                     default=1,
    #                     required=False,
    #                     help='Specify beam width')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    client_util = httpclient if FLAGS.protocol == "http" else grpcclient

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    merge_file = os.path.join(FLAGS.datasets_dir, "gpt2-merges.txt")
    vocab_file = os.path.join(FLAGS.datasets_dir, "gpt2-vocab.json")
    lambada_dataset_file = os.path.join(
        FLAGS.datasets_dir, "lambada_test.jsonl")
    enc = encoder.get_encoder(vocab_file, merge_file)

    all_ids, raw_text = load_data(enc, lambada_dataset_file, FLAGS.number_of_samples)
    prev_index = 0
    correct_num = 0
    # Only compare the last token
    while prev_index < len(all_ids):
        input_start_ids = all_ids[prev_index: prev_index + FLAGS.batch_size]
        context = [ids[:-1] for ids in input_start_ids]
        labels = [ids[-1:] for ids in input_start_ids]
        labels = np.asarray(labels)
        prev_index += FLAGS.batch_size
        input_len = np.array([[sentence.shape[-1]]
                             for sentence in context], np.uint32)

        padded_context = pad_sequence(
            context, batch_first=True, padding_value=END_ID)
        padded_context = padded_context.cpu().numpy().astype(np.uint32).reshape(
            [padded_context.shape[0], 1, padded_context.shape[1]])
        batch_size = padded_context.shape[0]

        # tile for beam search
        # padded_context = np.tile(padded_context, (1, FLAGS.beam_width, 1))
        # input_len = np.tile(input_len.reshape([-1, 1]), (1, FLAGS.beam_width)).reshape([-1, 1])
        # input_len = input_len.reshape((batch_size, FLAGS.beam_width))
        output_len = np.ones_like(input_len).astype(np.uint32)
        output_ids = send_requests(FLAGS.url, padded_context,
                                   input_len, output_len, FLAGS.verbose, 2, FLAGS.model_name)

        generated_tokens = output_ids[:, 0, -1]
        correct_num += (generated_tokens == labels).astype(np.int32).sum()

    prev_index = 0
    error_num = np.zeros([MAX_TEST_GRAM])
    total_num = np.zeros([MAX_TEST_GRAM])
    # Compare n-gram with all possible context
    # Require longer time
    while prev_index < len(all_ids) and FLAGS.n_gram_disabled is False:
        input_start_ids = all_ids[prev_index: prev_index + FLAGS.batch_size]
        prev_index += FLAGS.batch_size
        input_len = np.array([[sentence.shape[-1]]
                             for sentence in input_start_ids], np.uint32)

        padded_input_start_ids = pad_sequence(
            input_start_ids, batch_first=True, padding_value=END_ID)
        batch_size = padded_input_start_ids.shape[0]

        for i in range(padded_input_start_ids.shape[-1] - 1):

            context = padded_input_start_ids[:, :i+1]
            context = context.cpu().numpy().astype(np.uint32).reshape(
                [context.shape[0], 1, context.shape[1]])
            context_input_len = np.ones([batch_size], dtype=np.uint32) * (i+1)
            context_input_len = context_input_len.reshape([-1, 1])
            output_len = np.ones_like(context_input_len).astype(
                np.uint32) * MAX_TEST_GRAM

            output_ids = send_requests(FLAGS.url, context,
                                       context_input_len, output_len, FLAGS.verbose, 2, FLAGS.model_name)

            for j in range(1, MAX_TEST_GRAM + 1):
                if i + j < padded_input_start_ids.shape[1]:
                    generated_tokens = output_ids[:, :, i+1:i+1+j]
                    generated_tokens = generated_tokens.reshape([-1, j])
                    labels = padded_input_start_ids[:, i+1:i+1+j].numpy()
                    is_same = generated_tokens == labels
                    is_same = is_same.all(axis=-1)

                    # mask the token which is larger than input_len by True because we want to compute error num
                    mask = i+1+j > input_len
                    mask = mask.reshape([-1])
                    is_same = np.logical_or(is_same, mask)
                    is_diff = ~is_same
                    error_num[j - 1] += is_diff.astype(np.int32).sum()
                    total_num[j - 1] += (i+1+j <=
                                         input_len).astype(np.int32).sum()

    res = {}
    res["total_num_sent"] = len(all_ids)
    res["correct_num_last_token_pred"] = correct_num
    res["last_token_accuracy"] = "{:5.2f}".format(
        correct_num / len(all_ids) * 100)
    print("[INFO] last token accuracy: {}% (total token num: {})".format(
        res["last_token_accuracy"], len(all_ids)))

    if FLAGS.n_gram_disabled is False:
        accuracy = (total_num - error_num) / total_num * 100
        print("[INFO] accuracy under {} sentencs".format(len(all_ids)))
        for i in range(MAX_TEST_GRAM):
            res[f"{i+1}-gram_accuracy"] = "{:5.2f}".format(accuracy[i])
            res[f"{i+1}-gram_count"] = total_num[i]
            print("  {}-gram accuracy: {}% (total token num: {})".format(i +
                1, res[f"{i+1}-gram_accuracy"], res[f"{i+1}-gram_count"]))

    # Dump to csv
    with open(FLAGS.output_csv, mode='w') as csv_file:
        fieldnames = res.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(res)
