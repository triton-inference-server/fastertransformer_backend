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
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")

from transformers import PreTrainedTokenizerFast
from transformers import T5Tokenizer # transformers-4.10.0-py3
from utils.recover_bpe import recover_bpe
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
from sacrebleu import corpus_bleu

def create_inference_server_client(protocol, url, concurrency, verbose):
    client_util = httpclient if protocol == "http" else grpcclient
    if protocol == "http":
        return client_util.InferenceServerClient(url,
                                                concurrency=concurrency,
                                                verbose=verbose)
    elif protocol == "grpc":
        return client_util.InferenceServerClient(url,
                                                verbose=verbose)

def bleu_score(pred, ref):
    bleu = corpus_bleu(pred, [ref], force=True)
    print("       bleu score: {:6.2f}".format(bleu.score))
    print("       bleu counts: {}".format(bleu.counts))
    print("       bleu totals: {}".format(bleu.totals))
    print("       bleu precisions: {}".format(bleu.precisions))
    print("       bleu sys_len: {}; ref_len: {}".format(bleu.sys_len, bleu.ref_len))
    return bleu

def prepare_tensor(name, input, protocol):
    client_util = httpclient if protocol == "http" else grpcclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

class TranslationResult(object):
    def __init__(self, name, frame_work):
        self.name = name
        self.frame_work = frame_work # FT or HF
        self.file_name = name + ".txt"

        self.token_list = []
        self.batch_ids_list = []
        self.batch_seq_len_list = []
        self.batch_num = 0
        self.execution_time = 0.0  # seconds
        self.sentence_num = 0
        self.token_num = 0
        self.bleu_score = None
            
def translate(args_dict):
    torch.set_printoptions(precision=6)
    batch_size = args_dict['batch_size']
    source_file = args_dict["source"]
    tgt_file = args_dict["target"]
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    maximum_output_length = args_dict['maximum_output_length']
    max_ite = args_dict['max_iteration']

    tokenizer = T5Tokenizer.from_pretrained(args_dict['model'])
    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(args_dict['model'])

    with open(source_file, 'r') as f:
        src_text = recover_bpe(f.readlines())
        src_text = ["translate English to German: " + line.strip() for line in src_text]

    with open(tgt_file, 'r') as f:
        tgt_text = recover_bpe(f.readlines())

    translation_result_list = []
    translation_result_list.append(TranslationResult("ft_triton_warmup", "FT"))
    translation_result_list.append(TranslationResult("ft_triton", "FT"))
    for i in range(len(translation_result_list)):
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
            results = []
            ite_idx = 0
            while prev < len(src_text):
                input_texts = src_text[prev:prev+batch_size]
                prev += batch_size
                input_token = tokenizer(input_texts, return_tensors='pt', padding=True)
                
                input_ids = input_token.input_ids.numpy().astype(np.uint32)
                mem_seq_len = torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
                mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])
                runtime_top_k = (topk * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                runtime_top_p = topp * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                beam_search_diversity_rate = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                temperature = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                len_penalty = 0.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                repetition_penalty = 1.0 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                random_seed = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint64)
                is_return_log_probs = True * np.ones([input_ids.shape[0], 1]).astype(bool)
                max_output_len = (maximum_output_length * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                bad_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
                stop_words_ids = np.array([[[0], [-1]]] * input_ids.shape[0], dtype=np.int32)
                beam_width = (args_dict['beam_width'] * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
                start_ids = 0 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
                end_ids = 1 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

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

                # factual-nucleus arguments
                # top_p_decay = 0.9 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                # top_p_min = 0.5 * np.ones([input_ids.shape[0], 1]).astype(np.float32)
                # top_p_reset_ids = 13 * np.ones([input_ids.shape[0], 1]).astype(np.uint32)
                # inputs.append(prepare_tensor("top_p_decay", top_p_decay, args_dict["protocol"]))
                # inputs.append(prepare_tensor("top_p_min", top_p_min, args_dict["protocol"]))
                # inputs.append(prepare_tensor("top_p_reset_ids", top_p_reset_ids, args_dict["protocol"]))
                
                print("set request")
                result = client.infer(model_name, inputs)
                print("get request")
                results.append(result)
                ite_idx += 1
                if ite_idx >= max_ite:
                    break
                
            for result in results:
                ft_decoding_outputs = result.as_numpy("output_ids")
                ft_decoding_seq_lens = result.as_numpy("sequence_length")
                cum_log_probs = result.as_numpy("cum_log_probs")
                output_log_probs = result.as_numpy("output_log_probs")

                translation_result_list[i].batch_ids_list.append(ft_decoding_outputs)
                translation_result_list[i].batch_seq_len_list.append(ft_decoding_seq_lens)
                
                translation_result_list[i].sentence_num += len(input_token)
                translation_result_list[i].batch_num += 1

        stop_time = datetime.now()
        translation_result_list[i].execution_time = (stop_time - start_time).total_seconds()
        
        for batch_token, batch_seq_len in zip(translation_result_list[i].batch_ids_list, translation_result_list[i].batch_seq_len_list):
            for j in range(len(batch_token)):
                translation_result_list[i].token_list.append(fast_tokenizer.decode(batch_token[j][0][:batch_seq_len[j][0]], skip_special_tokens=True))
                translation_result_list[i].token_num += batch_seq_len[j][0]

        translation_result_list[i].bleu_score = bleu_score(translation_result_list[i].token_list, tgt_text[:len(translation_result_list[i].token_list)])
        with open(translation_result_list[i].name + ".txt", 'w') as f:
            for line in translation_result_list[i].token_list:
                f.write(line)
    
    for t in translation_result_list:
        if t.name.find("warmup") != -1: 
            continue
        print(f"{t.name} translates {t.batch_num} batches taking {t.execution_time:.2f} sec to translate "
                f"{t.token_num} tokens, BLEU score: {t.bleu_score.score:.2f}, {(t.token_num / t.execution_time):.0f} tokens/sec."
                f" ({t.bleu_score.sys_len} words, {(t.bleu_score.sys_len / t.execution_time):.0f} words/sec)")
    
        if args_dict["BLEU_threshold"] != None:
            assert t.bleu_score.score >= args_dict["BLEU_threshold"], f"[ERROR] {t.name} test fail!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-batch', '--batch_size', type=int, default=1, metavar='NUMBER',
                        help='batch size (default: 1)')
    parser.add_argument('-max_output_len', '--maximum_output_length', type=int, default=128, metavar='NUMBER',
                        help='maximum output length (default: 128)')
    parser.add_argument("--source", default="tools/t5_utils/test.en",
                        help="Path to the source file.")
    parser.add_argument("--target", default="tools/t5_utils/test.de",
                        help="Path to the target file.")
    parser.add_argument('-model', '--model', type=str, default="t5-small", metavar='STRING',
                        help='T5 model size.', choices=["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"])
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-i',
                        '--protocol',
                        type=str,
                        required=False,
                        default='http',
                        help='Protocol ("http"/"grpc") used to ' +
                        'communicate with inference service. Default is "http".')
    parser.add_argument('--BLEU_threshold', type=float,
                        help='Threshold of FT BLEU score')
    parser.add_argument('-max_ite', '--max_iteration', type=int, default=100000, metavar='NUMBER',
                        help='Maximum iteraiton for translation, default is 100000 (as large as possible to run all test set).')
    args = parser.parse_args()

    translate(vars(args))
