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

import argparse
import configparser
import dataclasses
import json
import os
import pathlib
import time
import json

import numpy as np
import torch
import torch.distributed as dist
import typing
from tqdm import tqdm

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

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

def preds_to_text(tokenizer, preds):
    preds = preds.cpu().numpy().tolist()
    preds = [pred[0] for pred in preds]

    preds_text = []
    for _, pred in enumerate(preds):
        if tokenizer.eos_id in pred:
            idx = pred.index(tokenizer.eos_id)
            pred = pred[:idx]

        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(tokenizer, 'special_token_to_id'):
            pred = [id for id in pred if id not in tokenizer.special_token_to_id.values()]
        pred = tokenizer.ids_to_text(pred)
        preds_text.append(pred)

    return preds_text

def accuracy_score(pred, ref):
    assert len(pred) == len(ref)
    total = len(pred)
    correct = 0
    for p, r in zip(pred, ref):
        if p == r:
            correct += 1
        # else:
        #     print(f"[pred]: {p} [label]: {r}")
    accuracy = correct / total
    print(f"[accuracy]: {accuracy}")
    return accuracy

@dataclasses.dataclass
class Metric:
    acc: float

@dataclasses.dataclass
class RequestAndResult:
    model_answer: str
    target: str
    metrics: Metric

def mnli_task(args_dict):
    data_path = args_dict["data_path"]
    boolq_test_data = []
    with open(data_path, "r") as f:
        f_list = list(f)
        for json_str in f_list:
            data = json.loads(json_str)
            boolq_test_data.append(data)

    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    beam_size = args_dict['beam_width']
    max_output_len = args_dict['max_output_len']
    beam_search_diversity_rate = args_dict['beam_search_diversity_rate']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    return_output_log_probs = args_dict["return_output_log_probs"]

    if args_dict['ckpt_path'] is None:
        raise Exception("Megatron T5 model needs to specify checkpoint path !")

    ckpt_path = pathlib.Path(args_dict['ckpt_path'])
    ## read checkpoint config if exists
    ckpt_config = configparser.ConfigParser()

    vocab_path = ckpt_path / "vocab.txt"
    ckpt_config_path = ckpt_path / "config.ini"
    if ckpt_config_path.is_file():
        ckpt_config.read(ckpt_config_path)
    else:
        raise Exception("config file does exist with the ckpt !")

    print("\n=============== Argument ===============")
    for key in args_dict:
        print("{}: {}".format(key, args_dict[key]))
    print("========================================")

    ## build tokenizer, dataset, dataloader
    tokenizer_t5 = get_nmt_tokenizer(
                    library='megatron',
                    model_name='BertWordPieceCase',
                    tokenizer_model=None,
                    vocab_file=vocab_path.as_posix(),
                    merges_file=None,
                    legacy=False,
                    )

    assert tokenizer_t5.bos_id == ckpt_config.getint("decoder", "decoder_start_token_id")
    assert tokenizer_t5.eos_id == ckpt_config.getint("decoder", "eos_token_id")

    token_params = {
        tokenizer_t5.bos_token: tokenizer_t5.bos_id,
        tokenizer_t5.eos_token: tokenizer_t5.eos_id,
        tokenizer_t5.pad_token: tokenizer_t5.pad_id,
    }
    print(f"tokenizer special tokens: {token_params}")

    prompt_start_id = ckpt_config.getint("encoder", "prompt_learning_start_id")
    prompt_length = ckpt_config.getint("task_0", "prompt_length")
    prompt_ids = torch.IntTensor(np.array([range(prompt_start_id, prompt_start_id+prompt_length, 1)])).to(torch.int32)

    preds_list = []
    labels_list = []
    results_list = []

    # Load prompt
    prompt_weights_dict = {}
    prompt_length_dict = {}
    if args_dict["use_request_prompt_embedding"]:
        d_model = ckpt_config.getint("encoder", "d_model")
        num_tasks = ckpt_config.getint("encoder", "num_tasks")
        for i in range(num_tasks):
            task_name = ckpt_config.get(f"task_{i}", "task_name")
            prompt_length_dict[task_name] = ckpt_config.getint(f"task_{i}", "prompt_length")
            prompt_weights_dict[task_name] = np.fromfile(ckpt_path / f"model.prompt_table.{task_name}.weight.bin", dtype=np.float32).reshape([prompt_length_dict[task_name], d_model])

    start = time.time()
    url = "localhost:8000" if args_dict["protocol"] == "http" else "localhost:8001"
    model_name = "fastertransformer"
    request_parallelism = 10
    verbose = False
    with create_inference_server_client(args_dict["protocol"],
                                        url,
                                        concurrency=request_parallelism,
                                        verbose=verbose) as client:

        results = []
        prev_idx = 0
        current_idx = 0
        while current_idx < len(boolq_test_data):
            current_idx += batch_size

            batch_boolq_test_data = boolq_test_data[prev_idx:current_idx]
            max_length = 0
            lengths = []
            batch_tokens = []
            labels = []
            for data in batch_boolq_test_data:
                text = data['passage'] + data['question']
                batch_tokens.append(tokenizer_t5.text_to_ids(text))

                # Need to append these two ids. These diff may be caused by tokenizer
                batch_tokens[-1].append(28996)
                batch_tokens[-1].append(102)
                max_length = max(len(batch_tokens[-1]), max_length)
                lengths.append(len(batch_tokens[-1]))
                labels.append(data['answer'])

            for i in range(len(batch_tokens)):
                batch_tokens[i] = torch.cat((prompt_ids,
                                            torch.IntTensor([batch_tokens[i]]).to(torch.int32),
                                            tokenizer_t5.eos_id * torch.ones([1, max_length - len(batch_tokens[i])]).to(torch.int32)), axis=1)
                lengths[i] += prompt_ids.shape[1]
            
            batch_tokens_tensor = torch.cat(batch_tokens)
            lengths_tensor = torch.IntTensor(np.array(lengths)).to(torch.int32)
            
            input_ids = batch_tokens_tensor.numpy().astype(np.uint32)
            mem_seq_len = lengths_tensor.numpy().astype(np.uint32).reshape([-1, 1])
            runtime_top_k = (topk * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            runtime_top_p = topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            random_seed = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint64)
            is_return_log_probs = return_output_log_probs * np.ones([input_ids.shape[0], 1]).astype(bool)
            max_output_len_tensor = (max_output_len * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            stop_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            beam_width = (args_dict['beam_width'] * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            # start_ids = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            # end_ids = 1 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            
            prompt_learning_task_name_ids = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            
            task_names = ["boolq" for i in range(input_ids.shape[0])]
            if args_dict["use_request_prompt_embedding"]:
                request_prompt_lengths = np.array([prompt_length_dict[t] for t in task_names]).astype(np.uint32).reshape([-1, 1])
                request_prompt_embedding = np.array([prompt_weights_dict[t] for t in task_names]).astype(np.float16)

            inputs = [
                prepare_tensor("input_ids", input_ids, args_dict["protocol"]),
                prepare_tensor("sequence_length", mem_seq_len, args_dict["protocol"]),
                prepare_tensor("runtime_top_k", runtime_top_k, args_dict["protocol"]),
                prepare_tensor("runtime_top_p", runtime_top_p, args_dict["protocol"]),
                prepare_tensor("beam_search_diversity_rate", beam_search_diversity_rate, args_dict["protocol"]),
                prepare_tensor("temperature", temperature, args_dict["protocol"]),
                prepare_tensor("len_penalty", len_penalty, args_dict["protocol"]),
                prepare_tensor("repetition_penalty", repetition_penalty, args_dict["protocol"]),
                prepare_tensor("random_seed", random_seed, args_dict["protocol"]),
                prepare_tensor("is_return_log_probs", is_return_log_probs, args_dict["protocol"]),
                prepare_tensor("max_output_len", max_output_len_tensor, args_dict["protocol"]),
                prepare_tensor("beam_width", beam_width, args_dict["protocol"]),
                # prepare_tensor("start_id", start_ids, args_dict["protocol"]),
                # prepare_tensor("end_id", end_ids, args_dict["protocol"]),
                prepare_tensor("bad_words_list", bad_words_ids, args_dict["protocol"]),
                prepare_tensor("stop_words_list", stop_words_ids, args_dict["protocol"]),
            ]
            if args_dict["use_request_prompt_embedding"]:
                inputs.append(prepare_tensor("request_prompt_lengths", request_prompt_lengths, args_dict["protocol"]))
                inputs.append(prepare_tensor("request_prompt_embedding", request_prompt_embedding, args_dict["protocol"]))
            else:
                inputs.append(prepare_tensor("prompt_learning_task_name_ids", prompt_learning_task_name_ids, args_dict["protocol"]))
            
            print("set request")
            result = client.infer(model_name, inputs)
            print("get request")
            results.append(result)
            labels_list += labels

            prev_idx = current_idx

        for result in results:
            ft_decoding_outputs = result.as_numpy("output_ids")
            ft_decoding_seq_lens = result.as_numpy("sequence_length")
            # cum_log_probs = result.as_numpy("cum_log_probs")
            # output_log_probs = result.as_numpy("output_log_probs")
            preds = preds_to_text(tokenizer_t5, torch.IntTensor(ft_decoding_outputs))
            preds_list += preds

        results_list.extend([
            RequestAndResult(
                model_answer=pred,
                target=label,
                metrics=Metric(acc=pred == label)
            )
            for pred, label in zip(preds_list, labels_list)
        ])

    end = time.time()
    print(f"\n[Elapsed Time]: {end - start} seconds")

    accuracy = accuracy_score(preds_list, labels_list)
    output_path = args_dict.get("output_path")
    if output_path is not None:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            results = {
                "results": {
                    "mnli": {
                        "acc": accuracy
                    }
                },
                "output": {
                    "mnli": [
                        dataclasses.asdict(r) for r in results_list
                    ]
                }
            }
            json.dump(results, output_file)
    if args_dict["accuracy_threshold"] != None:
        assert args_dict["accuracy_threshold"] >= accuracy, f"[ERROR] boolq test fail!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='beam width (default: 1)')
    parser.add_argument('-s', '--max_output_len', type=int, default=10, metavar='NUMBER',
                        help='max output length (default: 10)')
    parser.add_argument('-diversity_rate', '--beam_search_diversity_rate', type=float, default=0.0, metavar='NUMBER',
                        help='deviersity rate of beam search. default is 0. When diversity rate = 0, it is equivalent to the naive beams earch.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-data_path', '--data_path', type=str, help="the MNLI task data path", default="./tools/t5_utils/boolq_test.jsonl")
    parser.add_argument('--ckpt_path', type=str, help='path to the checkpoint file.', required=True)
    parser.add_argument('--output_path', help='path to results file with calculated metrics.')
    parser.add_argument('--return_output_log_probs', action='store_true',
                        help='Return the log probability of generated tokens.')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('--accuracy_threshold', type=float,
                        help='Threshold of FT accuracy score')
    parser.add_argument('-use_request_prompt_embedding', '--use_request_prompt_embedding', action="store_true",
                        help = "If given, inference will use prompt embedding, not task_name_ids")
    args = parser.parse_args()

    mnli_task(vars(args))