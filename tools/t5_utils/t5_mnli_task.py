# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import sys
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import time
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")
sys.path.append(dir_path + "/../../../NeMo")

from utils.recover_bpe import recover_bpe
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from omegaconf.omegaconf import OmegaConf, open_dict
from nemo.collections.nlp.data.glue_benchmark.glue_benchmark_dataset import (
    TextToTextGLUEDataset,
    TextToTextXNLIDataset,
)
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer

def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)

def _build_dataset(data_cfg, tokenizer):
    if data_cfg.task_name == 'xnli':
        dataset = TextToTextXNLIDataset(
            data_cfg.file_path,
            task_name=data_cfg.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_cfg.max_seq_length,
            lang_list=data_cfg.eval_languages,
        )
    else:
        dataset = TextToTextGLUEDataset(
            data_cfg.file_path,
            task_name=data_cfg.task_name,
            tokenizer=tokenizer,
            max_seq_length=data_cfg.max_seq_length,
        )
    return dataset

def preds_and_labels_to_text(tokenizer, preds, labels):
    preds = preds.cpu().numpy().tolist()
    labels = labels.cpu().numpy().tolist()
    preds = [pred[0] for pred in preds]

    preds_text, labels_text = [], []
    for _, (pred, label) in enumerate(zip(preds, labels)):
        if tokenizer.eos_id in pred:
            idx = pred.index(tokenizer.eos_id)
            pred = pred[:idx]

        # Legacy sentencepiece detokenization still preserves special tokens which messes up exact string match.
        if hasattr(tokenizer, 'special_token_to_id'):
            pred = [id for id in pred if id not in tokenizer.special_token_to_id.values()]
            label = [id for id in label if id not in tokenizer.special_token_to_id.values()]
        pred = tokenizer.ids_to_text(pred)
        label = tokenizer.ids_to_text(label)
        preds_text.append(pred)
        labels_text.append(label)

    return preds_text, labels_text

def accuracy_score(pred, ref):
    assert len(pred) == len(ref)
    total = len(pred)
    correct = 0
    for p, r in zip(pred, ref):
        if p == r:
            correct += 1
        # else:
        #     print(f"[pred]: {p} [label]: {r}")
    print(f"[accuracy]: {correct / total}")
    return correct / total

class InputToken:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def mnli_task(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    maximum_output_length = args_dict['maximum_output_length']

    ## build tokenizer, dataset, dataloader
    tokenizer_t5 = get_nmt_tokenizer(
                    library='megatron',
                    model_name='BertWordPieceCase',
                    tokenizer_model=None,
                    vocab_file=None,
                    merges_file=None,
                    legacy=False,
                    )
    mnli_cfg = OmegaConf.create({
        "file_path": args_dict['data_path'],
        "task_name": "mnli",
        "max_seq_length": 512
    })

    mnli_dataset = _build_dataset(mnli_cfg, tokenizer_t5)

    data_loader = torch.utils.data.DataLoader(
                mnli_dataset,
                collate_fn=mnli_dataset.collate_fn,
                batch_size=batch_size,
                num_workers=8,
                pin_memory=True,
                drop_last=True)

    sys.stdout.flush()
    
    url = "localhost:8000" if args_dict["protocol"] == "http" else "localhost:8001"
    model_name = "fastertransformer"
    request_parallelism = 10
    verbose = False
    with create_inference_server_client(args_dict["protocol"],
                                        url,
                                        concurrency=request_parallelism,
                                        verbose=verbose) as client:
        prev = 0
        start_time = datetime.now()
        preds_list = []
        labels_list = []
        for idx, batch in tqdm(enumerate(data_loader)):
            input_token = InputToken(batch['text_enc'], batch['enc_mask'])
            
            input_ids = input_token.input_ids.numpy().astype(np.uint32)
            mem_seq_len = torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
            mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])

            # TODO(bhsueh) should be set to optional inputs in the future
            runtime_top_k = (topk * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            runtime_top_p = topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            temperature = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            len_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            repetition_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
            random_seed = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint64)
            is_return_log_probs = False * np.ones([input_ids.shape[0], 1]).astype(bool)
            max_output_len = (maximum_output_length * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            stop_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
            beam_width = (args_dict['beam_width'] * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
            start_ids = 101 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
            end_ids = 102 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

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
                prepare_tensor("max_output_len", max_output_len, args_dict["protocol"]),
                prepare_tensor("beam_width", beam_width, args_dict["protocol"]),
                prepare_tensor("start_id", start_ids, args_dict["protocol"]),
                prepare_tensor("end_id", end_ids, args_dict["protocol"]),
                prepare_tensor("bad_words_list", bad_words_ids, args_dict["protocol"]),
                prepare_tensor("stop_words_list", stop_words_ids, args_dict["protocol"]),
            ]
            
            print("set request")
            result = client.infer(model_name, inputs)
            print("get request")
            ft_decoding_outputs = result.as_numpy("output_ids")
            preds, labels = preds_and_labels_to_text(tokenizer_t5, torch.IntTensor(ft_decoding_outputs), batch['labels'])
            # print(preds, labels)
            preds_list += preds
            labels_list += labels

    accuracy_score(preds_list, labels_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-max_output_len', '--maximum_output_length', type=int, default=10, metavar='NUMBER',
                        help='maximum output length (default: 10)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-data_path', '--data_path', type=str, required=True, help="the MNLI task data path")
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    args = parser.parse_args()

    mnli_task(vars(args))
