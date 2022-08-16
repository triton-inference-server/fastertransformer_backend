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
import os
import sys
from datetime import datetime
import numpy as np
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")
sys.path.append(dir_path + "/../../../NeMo")

from tqdm.auto import tqdm

from utils.gpt_prompt_learning import GPTPromptLearningTokenizer, get_prompt_dataset, IdentityTestPromptProcess

## triton client
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s | %(name)-7s | %(levelname)-8s: %(message)s"
)
logger = logging.getLogger(__name__)

SEQ={'context': 'In signal processing, data compression, source coding, or bit-rate reduction \
involves encoding information using fewer bits than the original representation. Compression can be either lossy or lossless. \
Lossless compression reduces bits by identifying and eliminating statistical redundancy. No information is lost in lossless compression. \
Lossy compression reduces bits by identifying unnecessary information and removing it. The process of reducing the size of a data file is referred to as data compression. \
In the context of data transmission, it is called source coding (encoding done at the source of the data before it is stored or transmitted) in opposition to channel coding.', \
'question': 'What involves encoding information using fewer bits than the original representation?'}

def prepare_tensor(name, input):
    client_util = httpclient
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

def squad_task(args_dict):
    torch.set_printoptions(precision=6)
    topk = args_dict['sampling_topk']
    topp = args_dict['sampling_topp']
    max_output_len = args_dict['max_output_len']
    virtual_prompt_model_path = args_dict['virtual_prompt_model_path']
    gpt_model_path = args_dict['gpt_model_path']

    prompt_tokenizer = GPTPromptLearningTokenizer(gpt_model_path, virtual_prompt_model_path)
    prompt_process = IdentityTestPromptProcess(
        tokenizer=prompt_tokenizer.tokenizer,
        virtual_prompt_source=prompt_tokenizer.virtual_prompt_source,
        task_templates=prompt_tokenizer.task_templates,
        pseudo_tokens=prompt_tokenizer.pseudo_tokens,
        pad_token_id=prompt_tokenizer.pad_token_id,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=False,
        for_train=True,
    )

    input_seqs = [SEQ]
    task_name = "squad"

    ## assume prompt inference data type is fp16
    use_request_prompt_embedding = args.use_request_prompt_embedding
    squad_prompt = np.ones([1,1],dtype=np.float16)
    prompt_length = 0
    # prompt type
    # no_prompt, <-- 0
    # soft_prompt, <-- 1
    # prefix_prompt, <-- 2
    # p_prompt_tuning <-- 3
    prompt_type = 3
    if use_request_prompt_embedding:
        squad_prompt = \
            torch.load(f"{virtual_prompt_model_path}/model_weights.ckpt")['prompt_table'][f'prompt_table.{task_name}.prompt_embeddings.weight'] \
            .detach().cpu().numpy().astype(np.float16)
        prompt_length = squad_prompt.shape[0]

    client_util = httpclient
    
    sys.stdout.flush()
    
    url = "localhost:8000"
    model_name = "fastertransformer"
    request_parallelism = 1
    verbose = False

    with client_util.InferenceServerClient(url,
                                            concurrency=request_parallelism,
                                            verbose=verbose) as client:
        taskname_ids, input_start_ids, input_length = prompt_process.encode(input_seqs, task_name)

        input_start_ids = np.array(input_start_ids).astype(np.uint32)
        input_length = np.array([[i] for i in input_length]).astype(np.uint32)
        output_len = np.ones_like(input_length).astype(np.uint32) * max_output_len

        # TODO(bhsueh) should be set to optional inputs in the future
        runtime_top_k = topk * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        runtime_top_p = topp * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        beam_search_diversity_rate = 0.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        temperature = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        len_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        repetition_penalty = 1.0 * np.ones([input_start_ids.shape[0], 1]).astype(np.float32)
        random_seed = 0 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint64)
        is_return_log_probs = True * np.ones([input_start_ids.shape[0], 1]).astype(np.bool)
        beam_width = (1 * np.ones([input_start_ids.shape[0], 1])).astype(np.uint32)
        start_ids = 50256 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        end_ids = 50256 * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        bad_words_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
            np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)
        stop_word_list = np.concatenate([np.zeros([input_start_ids.shape[0], 1, 1]).astype(
            np.int32), (-1 * np.ones([input_start_ids.shape[0], 1, 1])).astype(np.int32)], axis=1)
        request_prompt_embedding = np.repeat(np.expand_dims(squad_prompt,axis=0), input_start_ids.shape[0], axis=0)
        request_prompt_lengths = prompt_length * \
            np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        request_prompt_type = prompt_type * \
            np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        prompt_learning_task_name_ids = np.array([[task_name_id] for task_name_id in taskname_ids], dtype=np.uint32)

        inputs = [
            prepare_tensor("input_ids", input_start_ids),
            prepare_tensor("input_lengths", input_length),
            prepare_tensor("request_output_len", output_len),
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
            prepare_tensor("bad_words_list", bad_words_list),
            prepare_tensor("stop_words_list", stop_word_list)
        ]

        if use_request_prompt_embedding:
            inputs.append(prepare_tensor("request_prompt_embedding", request_prompt_embedding))
            inputs.append(prepare_tensor("request_prompt_lengths", request_prompt_lengths))
            inputs.append(prepare_tensor("request_prompt_type", request_prompt_type))
        else:
            inputs.append(prepare_tensor("prompt_learning_task_name_ids", prompt_learning_task_name_ids))
        
        print("set request")
        result = client.infer(model_name, inputs)
        print("get request")
        output_data = result.as_numpy("output_ids")
        output_data = output_data.reshape([-1, output_data.shape[-1]])
        np.savetxt("triton_out", output_data, fmt='%u')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-max_output_len', type=int, default=10, metavar='NUMBER',
                        help='maximum output length (default: 10)')
    parser.add_argument('-beam', '--beam_width', type=int, default=1, metavar='NUMBER',
                        help='Beam width for beam search. If setting 1, then using sampling.')
    parser.add_argument('-topk', '--sampling_topk', type=int, default=1, metavar='NUMBER',
                        help='Candidate (k) value of top k sampling in decoding. Default is 1.')
    parser.add_argument('-topp', '--sampling_topp', type=float, default=0.0, metavar='NUMBER',
                        help='Probability (p) value of top p sampling in decoding. Default is 0.0. ')
    parser.add_argument('-virtual_prompt_model_path', '--virtual_prompt_model_path', type=str, required=True, help="the extracted virtual_prompt_model path")
    parser.add_argument('-gpt_model_path', '--gpt_model_path', type=str, required=True, help="the extracted gpt model path")
    parser.add_argument('-use_request_prompt_embedding', '--use_request_prompt_embedding', action="store_true",
                        help = "If given, inference will use prompt embedding, not task_name_ids")
    args = parser.parse_args()

    squad_task(vars(args))