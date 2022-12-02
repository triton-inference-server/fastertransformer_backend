#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import csv

import torch
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from tritonclient.utils import np_to_triton_dtype

import tools.utils.gpt_token_encoder as encoder

GPT_START_ID = 50256
GPT_END_ID = 50256

MAX_TEST_GRAM = 4


def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)


def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t


def send_requests(url, input_start_ids, input_len, output_len, verbose, flags, model_name="fastertransformer"):
    with create_inference_server_client(flags.protocol,
                                        url,
                                        1,
                                        verbose=verbose) as client:
        input_data = input_start_ids
        runtime_top_k = (flags.topk * np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
        runtime_top_p = flags.topp * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = True * np.ones([input_start_ids.shape[0], 1]).astype(np.bool)
        beam_width = (flags.beam_width * np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
        start_ids = 50256 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        end_ids = 50256 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        bad_words_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)
        stop_word_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)

        inputs = [
                prepare_tensor("input_ids", input_data, flags.protocol),
                prepare_tensor("input_lengths", input_len, flags.protocol),
                prepare_tensor("request_output_len", output_len, flags.protocol),
                prepare_tensor("runtime_top_k", runtime_top_k, flags.protocol),
                prepare_tensor("runtime_top_p", runtime_top_p, flags.protocol),
                prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, flags.protocol),
                prepare_tensor("temperature", temperature, flags.protocol),
                prepare_tensor("len_penalty", len_penalty, flags.protocol),
                prepare_tensor("repetition_penalty", repetition_penalty, flags.protocol),
                prepare_tensor("random_seed", random_seed, flags.protocol),
                prepare_tensor("is_return_log_probs", is_return_log_probs, flags.protocol),
                prepare_tensor("beam_width", beam_width, flags.protocol),
                prepare_tensor("start_id", start_ids, flags.protocol),
                prepare_tensor("end_id", end_ids, flags.protocol),
                prepare_tensor("bad_words_list", bad_words_list, flags.protocol),
                prepare_tensor("stop_words_list", stop_word_list, flags.protocol),
        ]
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_len)
        inputs[2].set_data_from_numpy(output_len)
        inputs[3].set_data_from_numpy(runtime_top_k)
        inputs[4].set_data_from_numpy(runtime_top_p)
        inputs[5].set_data_from_numpy(beam_search_diversity_rate)
        inputs[6].set_data_from_numpy(temperature)
        inputs[7].set_data_from_numpy(len_penalty)
        inputs[8].set_data_from_numpy(repetition_penalty)
        inputs[9].set_data_from_numpy(random_seed)
        inputs[10].set_data_from_numpy(is_return_log_probs)
        inputs[11].set_data_from_numpy(beam_width)
        inputs[12].set_data_from_numpy(start_ids)
        inputs[13].set_data_from_numpy(end_ids)
        inputs[14].set_data_from_numpy(bad_words_list)
        inputs[15].set_data_from_numpy(stop_word_list)
        result = client.infer(model_name, inputs)

        return result.as_numpy("output_ids")


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
    parser.add_argument('--n-gram-enabled',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable n-gram calculation')
    parser.add_argument('--number-of-samples',
                        type=int,
                        required=False,
                        default=None,
                        help='Limits number of samples for test')
    parser.add_argument('-beam',
                        '--beam_width',
                        type=int,
                        default=1,
                        required=False,
                        help='Specify beam width')
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
    parser.add_argument('--accuracy_threshold', type=float,
                        help='Threshold of FT accuracy score')
    parser.add_argument('--model-variant',
                        type=str,
                        default='gpt',
                        choices=['gpt', 'bloom'],
                        help='The type of GPT model variants. `gpt` indicates '
                             'the original GPT model.')
    parser.add_argument('--tokenizer-name-or-path',
                        type=str,
                        default=None,
                        help='HF tokenizer name or path. If None, the default '
                             'tokenizer will be used.')

    FLAGS = parser.parse_args()
    if (FLAGS.protocol != "http") and (FLAGS.protocol != "grpc"):
        print("unexpected protocol \"{}\", expects \"http\" or \"grpc\"".format(
            FLAGS.protocol))
        exit(1)

    if FLAGS.url is None:
        FLAGS.url = "localhost:8000" if FLAGS.protocol == "http" else "localhost:8001"

    lambada_dataset_file = os.path.join(FLAGS.datasets_dir, "lambada_test.jsonl")

    if FLAGS.model_variant == 'gpt':
        merge_file = os.path.join(FLAGS.datasets_dir, "gpt2-merges.txt")
        vocab_file = os.path.join(FLAGS.datasets_dir, "gpt2-vocab.json")
        enc = encoder.get_encoder(vocab_file, merge_file)
        start_id = GPT_START_ID
        end_id = GPT_END_ID
    elif FLAGS.model_variant == 'bloom':
        tokenizer_path = FLAGS.tokenizer_name_or_path or 'bigscience/bloom'
        enc = AutoTokenizer.from_pretrained(tokenizer_path)
        start_id = enc.bos_token_id
        end_id = enc.eos_token_id
    else:
        raise NotImplementedError(
            f'model_variant {FLAGS.model_variant} is not supported.')

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
            context, batch_first=True, padding_value=end_id)
        padded_context = padded_context.cpu().numpy().astype(np.uint32)
        batch_size = padded_context.shape[0]

        output_len = np.ones_like(input_len).astype(np.uint32)
        output_ids = send_requests(FLAGS.url, padded_context,
                                   input_len, output_len, FLAGS.verbose, FLAGS, FLAGS.model_name)

        generated_tokens = []
        for output_id, i_len in zip(output_ids, input_len):
            generated_tokens.append(output_id[0][i_len])
        correct_num += (generated_tokens == labels).astype(np.int32).sum()

    prev_index = 0
    error_num = np.zeros([MAX_TEST_GRAM])
    total_num = np.zeros([MAX_TEST_GRAM])
    # Compare n-gram with all possible context
    # Require longer time
    while prev_index < len(all_ids) and FLAGS.n_gram_enabled:
        input_start_ids = all_ids[prev_index: prev_index + FLAGS.batch_size]
        prev_index += FLAGS.batch_size
        input_len = np.array([[sentence.shape[-1]]
                             for sentence in input_start_ids], np.uint32)

        padded_input_start_ids = pad_sequence(
            input_start_ids, batch_first=True, padding_value=end_id)
        batch_size = padded_input_start_ids.shape[0]

        for i in range(padded_input_start_ids.shape[-1] - 1):

            context = padded_input_start_ids[:, :i+1]
            context = context.cpu().numpy().astype(np.uint32)
            context_input_len = np.ones([batch_size, 1], dtype=np.uint32) * (i+1)
            output_len = np.ones_like(context_input_len).astype(np.uint32) * MAX_TEST_GRAM
            output_ids = send_requests(FLAGS.url, context,
                                       context_input_len, output_len, FLAGS.verbose, FLAGS, FLAGS.model_name)

            for j in range(1, MAX_TEST_GRAM + 1):
                if i + j < padded_input_start_ids.shape[1]:
                    generated_tokens = output_ids[:, 0, i+1:i+1+j]
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
    last_token_accuracy = correct_num / len(all_ids) * 100
    res["last_token_accuracy"] = f"{last_token_accuracy:5.2f}"
    print("[INFO] last token accuracy: {}% (total token num: {})".format(
        res["last_token_accuracy"], len(all_ids)))

    if FLAGS.n_gram_enabled:
        accuracy = (total_num - error_num) / total_num * 100
        print("[INFO] accuracy under {} sentences".format(len(all_ids)))
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

    if FLAGS.accuracy_threshold is not None:
        assert last_token_accuracy >= FLAGS.accuracy_threshold, \
            f'[ERROR] gpt evaluate_lambada test fail '\
            f'({last_token_accuracy} < threshold {FLAGS.accuracy_threshold}).'
        print(f"[INFO] gpt evaluate_lambada test pass!")
