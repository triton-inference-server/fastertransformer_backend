# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This example is used to verify the correctess on summarization task. So, we don't
put benchmark testing in this example.
'''

from __future__ import print_function
import argparse
import numpy as np
import os
import torch
from datasets import load_dataset, load_metric
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from tqdm import tqdm
import configparser
import datetime
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_model_location', type=str,
                        default='/models/T5/HF/t5-base/c-models/')
    parser.add_argument('--hf_model_location', type=str,
                        default='/models/T5/HF/t5-base/')
    parser.add_argument('--disable_summarize', action='store_true')
    parser.add_argument('--test_hf', action='store_true')
    parser.add_argument('--test_ft', action='store_true')
    parser.add_argument('--data_type', type=str,
                        choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument("--cache_path", type=str,
                        default="/workdir/datasets/ccdv/")
    parser.add_argument("--max_ite", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument('--rougeLsum_threshold', type=float,
                        help='Threshold of FT rougeLsum score')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    args = parser.parse_args()
    args_dict = vars(args)

    disable_summarize = args.disable_summarize
    test_hf = args.test_hf
    test_ft = args.test_ft

    ft_model_location = args.ft_model_location
    hf_model_location = args.hf_model_location

    tokenizer = AutoTokenizer.from_pretrained(hf_model_location)
    tokenizer.pad_token = tokenizer.eos_token
    dataset_cnn = load_dataset(
        "ccdv/cnn_dailymail", '3.0.0', cache_dir=args.cache_path)

    if test_hf:
        start_time = datetime.datetime.now()
        if args.data_type == "fp32":
            model = T5ForConditionalGeneration.from_pretrained(
                hf_model_location, torch_dtype=torch.float32).cuda()
        elif args.data_type == "fp16":
            model = T5ForConditionalGeneration.from_pretrained(
                hf_model_location, torch_dtype=torch.float16).cuda()
        elif args.data_type == "bf16":
            model = T5ForConditionalGeneration.from_pretrained(
                hf_model_location, torch_dtype=torch.bfloat16).cuda()
        stop_time = datetime.datetime.now()
        print(
            f"[INFO] load HF model spend {(stop_time - start_time).total_seconds()} sec")

    if test_ft:
        ckpt_config = configparser.ConfigParser()

        ckpt_config_path = os.path.join(ft_model_location, 'config.ini')
        if os.path.isfile(ckpt_config_path):
            ckpt_config.read(ckpt_config_path)
        else:
            assert False, "[ERROR] This example only support loading model with FT format directly."

        encoder_config = T5Config(vocab_size=ckpt_config.getint("encoder", "vocab_size"),
                                  d_model=ckpt_config.getint(
                                      "encoder", "d_model"),
                                  d_kv=ckpt_config.getint("encoder", "d_kv"),
                                  d_ff=ckpt_config.getint("encoder", "d_ff"),
                                  num_layers=ckpt_config.getint(
                                      "encoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint(
                                      "encoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint(
                                      "encoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "encoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get(
                                      "encoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint(
                                      "encoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint(
                                      "encoder", "eos_token_id"),
                                  is_gated_act=ckpt_config.getboolean(
                                      "encoder", "is_gated_act", fallback=0),
                                  )
        decoder_config = T5Config(vocab_size=ckpt_config.getint("decoder", "vocab_size"),
                                  d_model=ckpt_config.getint(
                                      "decoder", "d_model"),
                                  d_kv=ckpt_config.getint("decoder", "d_kv"),
                                  d_ff=ckpt_config.getint("decoder", "d_ff"),
                                  num_layers=ckpt_config.getint(
                                      "decoder", "num_layers"),
                                  num_decoder_layers=ckpt_config.getint(
                                      "decoder", "num_decoder_layers"),
                                  num_heads=ckpt_config.getint(
                                      "decoder", "num_heads"),
                                  relative_attention_num_buckets=ckpt_config.getint(
                                      "decoder", "relative_attention_num_buckets_or_max_pos_seq_len"),
                                  feed_forward_proj=ckpt_config.get(
                                      "decoder", "feed_forward_proj"),
                                  pad_token_id=ckpt_config.getint(
                                      "decoder", "pad_token_id"),
                                  eos_token_id=ckpt_config.getint(
                                      "decoder", "eos_token_id"),
                                  decoder_start_token_id=ckpt_config.getint(
                                      "decoder", "decoder_start_token_id"),
                                  is_gated_act=ckpt_config.getboolean(
                                      "decoder", "is_gated_act", fallback=0),
                                  )

    if disable_summarize:
        top_k = 1
        output_len = args.max_seq_len
    else:
        top_k = 2
        output_len = args.max_seq_len

    def summarize_ft(datapoint):
        if not disable_summarize:
            line = "summarize: " + datapoint['article']
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_tokens = tokenizer(line, return_tensors='pt')

        url = "localhost:8000" if args_dict["protocol"] == "http" else "localhost:8001"
        model_name = "fastertransformer"
        request_parallelism = 10
        verbose = False
        with create_inference_server_client(args_dict["protocol"],
                                            url,
                                            concurrency=request_parallelism,
                                            verbose=verbose) as client:
            input_token = line_tokens

            input_ids = input_token.input_ids.numpy().astype(np.uint32)
            mem_seq_len = torch.sum(
                input_token.attention_mask, dim=1).numpy().astype(np.uint32)
            mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])

            # TODO(bhsueh) should be set to optional inputs in the future
            runtime_top_k = (
                top_k * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            runtime_top_p = 0.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.float32)
            random_seed = 0 * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint64)
            is_return_log_probs = True * \
                np.ones([input_ids.shape[0], 1]).astype(bool)
            max_output_len = (
                output_len * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            bad_words_ids = np.array(
                [[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            stop_words_ids = np.array(
                [[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            beam_width = (
                1 * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            start_ids = decoder_config.decoder_start_token_id * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            end_ids = encoder_config.eos_token_id * \
                np.ones([input_ids.shape[0], 1]).astype(np.uint32)

            inputs = [
                prepare_tensor("input_ids", input_ids, args_dict["protocol"]),
                prepare_tensor("sequence_length", mem_seq_len,
                               args_dict["protocol"]),
                prepare_tensor("runtime_top_k", runtime_top_k,
                               args_dict["protocol"]),
                prepare_tensor("runtime_top_p", runtime_top_p,
                               args_dict["protocol"]),
                prepare_tensor("beam_search_diversity_rate",
                               beam_search_diversity_rate, args_dict["protocol"]),
                prepare_tensor("temperature", temperature,
                               args_dict["protocol"]),
                prepare_tensor("len_penalty", len_penalty,
                               args_dict["protocol"]),
                prepare_tensor("repetition_penalty",
                               repetition_penalty, args_dict["protocol"]),
                prepare_tensor("random_seed", random_seed,
                               args_dict["protocol"]),
                prepare_tensor("is_return_log_probs",
                               is_return_log_probs, args_dict["protocol"]),
                prepare_tensor("max_output_len", max_output_len,
                               args_dict["protocol"]),
                prepare_tensor("beam_width", beam_width,
                               args_dict["protocol"]),
                prepare_tensor("start_id", start_ids, args_dict["protocol"]),
                prepare_tensor("end_id", end_ids, args_dict["protocol"]),
                prepare_tensor("bad_words_list", bad_words_ids,
                               args_dict["protocol"]),
                prepare_tensor("stop_words_list", stop_words_ids,
                               args_dict["protocol"]),
            ]

            result = client.infer(model_name, inputs)

            output = result.as_numpy("output_ids")
            ft_output_len = result.as_numpy("sequence_length")

        tokens = output[0][0]

        output_lines = tokenizer.decode(output[0][0][:ft_output_len[0][0]])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    def summarize_hf(datapoint):
        if not disable_summarize:
            line = "summarize: " + datapoint['article']
        else:
            line = datapoint['article']
        line = line.strip()
        line = line.replace(" n't", "n't")

        line_encoded = tokenizer.encode(line, return_tensors='pt')
        line_encoded = line_encoded.cuda()

        with torch.no_grad():
            output = model.generate(line_encoded,
                                    max_length=output_len + 1,
                                    top_k=top_k,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)

        tokens = output[0].cpu().numpy()
        output_lines = tokenizer.decode(output[0][1:])
        output_lines = ".".join(output_lines.split('.')[:4]) + "."
        return output_lines, tokens

    if disable_summarize:
        tokens = []
    else:
        metric_ft = load_metric("rouge")
        metric_hf = load_metric("rouge")

    if not disable_summarize:
        datapoint = dataset_cnn['test'][0]
        if test_ft:
            summary_ft, _ = summarize_ft(datapoint)
            print('---------------------------------------------------------')
            print('FT Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary_ft)
            print('---------------------------------------------------------')
            metric_ft.add_batch(predictions=[summary_ft], references=[
                                datapoint['highlights']])

        if test_hf:
            summary_hf, _ = summarize_hf(datapoint)
            print('---------------------------------------------------------')
            print('HF Generated : ')
            print(' Article : ', datapoint['article'])
            print('\n Highlights : ', datapoint['highlights'])
            print('\n Summary : ', summary_hf)
            print('---------------------------------------------------------')
            metric_hf.add_batch(predictions=[summary_hf], references=[
                                datapoint['highlights']])

    ft_time = 0.0
    hf_time = 0.0
    for data_point_idx in tqdm(range(1, 11490, int(11490 / args.max_ite))):
        try:
            datapoint = dataset_cnn['test'][data_point_idx]

            start_time = datetime.datetime.now()
            if test_ft:
                summary_ft, tokens_ft = summarize_ft(datapoint)
            stop_time = datetime.datetime.now()
            ft_time += (stop_time - start_time).total_seconds()

            if ((test_hf and not disable_summarize) or disable_summarize):
                start_time = datetime.datetime.now()
                summary_hf, tokens_hf = summarize_hf(datapoint)
                stop_time = datetime.datetime.now()
                hf_time += (stop_time - start_time).total_seconds()

            if not disable_summarize:
                if test_ft:
                    metric_ft.add_batch(predictions=[summary_ft], references=[
                                        datapoint['highlights']])
                if test_hf:
                    metric_hf.add_batch(predictions=[summary_hf], references=[
                                        datapoint['highlights']])
            else:
                tokens.append((tokens_ft, tokens_hf))
        except:
            print('Error with datapoint : ', data_point_idx)

    def compute_exact_match(tokens, n_tokens=[1, 10, 25, 50, 100, 150, 200, 250]):
        em_metrics = []
        for t in n_tokens:
            errors = 0
            total = 0
            for ft_tokens, hf_tokens in tokens:
                if len(ft_tokens) > t and len(hf_tokens) > t:
                    total = total + 1
                    if not np.array_equal(ft_tokens[:t], hf_tokens[:t]):
                        errors = errors + 1

            if total > 0:
                print(f"{t}-token exact match acc: {100*(1-errors/total):.2f}")
                em_metrics.append(1 - errors / total)
            else:
                em_metrics.append(np.nan)

        return em_metrics

    if not disable_summarize:
        if test_ft:
            computed_metrics_ft = metric_ft.compute()

        if test_hf:
            computed_metrics_hf = metric_hf.compute()

            print(f'Hugging Face (total latency: {hf_time} sec)')
            for key in computed_metrics_hf.keys():
                print(f'{key} : {computed_metrics_hf[key].mid[2]*100}')

        if test_ft:
            print(f'Faster Transformers (total latency: {ft_time} sec)')
            for key in computed_metrics_ft.keys():
                print(f'{key} : {computed_metrics_ft[key].mid[2]*100}')
            if args.rougeLsum_threshold != None:
                assert computed_metrics_ft["rougeLsum"].mid[2] * \
                    100 >= args.rougeLsum_threshold, "[INFO] TEST FAIL !"
                print(f"[INFO] TEST PASS !")
    else:
        em_metrics = compute_exact_match(tokens)


if __name__ == '__main__':
    main()
