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

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../")
sys.path.append(dir_path + "/../../../NeMo")

from tqdm.auto import tqdm
import re
from omegaconf import OmegaConf
from nemo.core import Dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.modules.common import (
    VirtualPromptPlaceholderToken,
    VirtualPromptSource,
    VirtualPromptStyle,
)

import json
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)-15s | %(name)-7s | %(levelname)-8s: %(message)s"
)
logger = logging.getLogger(__name__)

# Modified based on
# https://github.com/NVIDIA/NeMo/blob/e165f653d47c4faf89ecd97720803b8ef964a6ce/nemo/collections/nlp/data/language_modeling/megatron/gpt_prompt_learning_dataset.py#L28
class GPTPromptLearningDataset(Dataset):
    """
    The dataset class for prompt-tuning or p-tuning pretrained GPT models.
    """

    def __init__(
        self,
        datasets,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = False,
        for_train: bool = False,
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.examples = []

        assert self.min_seq_length <= max_seq_length, "Min sequence length should be less than or equal to max"
        assert self.max_seq_length > 0, "Max sequence length should be greater than 0"

        logger.info("Loading and tokenizing dataset ... ")

        # Datasets is just a list of json dicts
        if isinstance(datasets[0], dict):
            self.load_data(datasets)

        # Datasets are a list of file path strings to .json or .jsonl files
        elif isinstance(datasets[0], str):
            for path in datasets:
                dataset = open(path, 'r', encoding='utf-8')
                self.load_data(dataset)
        else:
            raise ValueError("Datasets must be a list of dicts or a list of filepath strings")

    def load_data(self, dataset):
        """
        Loads a dataset by filling in the task templates specified in the config file
        with the information from each training/inference example. Converts all input
        text into token ids. Also replaces the <|VIRTUAL_PROMPT_#|> placeholders in
        the task templates with the actual virtual prompt token ids.

        params:
            dataset: A list of json objects or a dictionary objects each
                     containing the information needed for a training example
        """
        skipped = 0

        for json_line in tqdm(dataset):

            # Read example dict or load the information for a single example from .json file
            if type(json_line) == dict:
                doc = json_line
            else:
                doc = json.loads(json_line)

            taskname = doc["taskname"]
            prompt_template = self.task_templates[taskname]["prompt_template"]
            prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
            total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
            virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
            truncation_field = self.task_templates[taskname]['truncate_field']
            answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
            answer_field = self.task_templates[taskname]["answer_field"]

            input_example = prompt_template

            self._input_sanity_checks(
                total_virtual_tokens,
                virtual_token_splits,
                prompt_template,
                prompt_template_fields,
                truncation_field,
                answer_only_loss,
                answer_field,
                doc,
            )

            # Format the input example according to the template
            input_example = self._insert_text_in_template(input_example, prompt_template_fields, doc)
            input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)
            input_ids = self.tokenizer.text_to_ids(input_example)

            # Add BOS/EOS if desired, adds EOS by default
            if self.add_bos:
                input_ids = [self.tokenizer.bos_id] + input_ids
            if self.add_eos:
                input_ids = input_ids + [self.tokenizer.eos_id]

            # Try to truncate input text to fit into the max sequence length
            if len(input_ids) > self.max_seq_length:
                input_ids = self._truncate_input(truncation_field, input_ids, taskname, doc)

            # Skip example if the final length doesn't fit length requirements even after truncation
            if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
                if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                    taskname_id = self.tokenizer.text_to_ids(taskname)

                elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
                    taskname_id = self.task_templates[taskname]["task_id_num"]

                answer_start_idx, answer_text_ids = self._find_answer_start(taskname, input_ids, answer_field, doc)

                self.examples.append((taskname_id, input_ids[:answer_start_idx], answer_start_idx, answer_text_ids))
            else:
                skipped += 1

        logging.info(f'Skipped {skipped} sentences, sequence length too short or too long even after truncation')

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):
        # Sanity check amount of virtual token
        assert total_virtual_tokens > 0, "There should be at least one virtual prompt token"
        assert (
            total_virtual_tokens < self.max_seq_length
        ), "virtual prompt tokens should not exceed max sequence length"

        # Make sure virtual token splits add up to the total number of virtual tokens
        assert (
            sum(virtual_token_splits) == total_virtual_tokens
        ), "Sum of prompt token split values must equal total number of prompt tokens"

        # Make sure number of virtual prompt locations match the number of virtual prompt splits
        assert prompt_template.count('<|VIRTUAL_PROMPT_') == len(
            virtual_token_splits
        ), "The number of '<|VIRTUAL_PROMPT_n|>' markers and the number of prompt token splits must match"

        # Check if input example has fields not present in template
        keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        assert (
            len(keys_not_in_template) == 0
        ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Check that answer field checks if answer_only_loss was set to True
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            if field in doc.keys():
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # If some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            else:
                input_example = input_example.replace('{' + field + '}', "")

        return input_example

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|virtual_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = len(input_ids) - self.max_seq_length
            field_text = doc[truncation_field]
            field_text = self._add_leading_space(taskname, truncation_field, field_text)

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]

            # Replace original text ids with truncated text ids
            field_start, field_end = find_subsequence_location(input_ids, field_text_ids)
            input_ids = input_ids[:field_start] + truncated_text_ids + input_ids[field_end + 1 :]

        return input_ids

    def _find_answer_start(self, taskname, input_ids, answer_field, doc):
        """ Find the token ids corresponding to the answer start, for loss masking purposes.
            Assumes the answer is always at the end of the prompt.
        """
        answer_text = doc[answer_field]
        answer_text = self._add_leading_space(taskname, answer_field, answer_text)
        answer_text_ids = self.tokenizer.text_to_ids(answer_text)
        num_answer_text_ids = len(answer_text_ids)

        if self.add_eos:
            num_answer_text_ids += 1

        answer_start_idx = len(input_ids) - num_answer_text_ids

        return answer_start_idx,answer_text_ids

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def collate_fn_custom(self, batch):
        taskname_id = list()
        input_ids = list()
        input_length = list()
        answer_text_ids = list()

        max_length = 0
        for b in batch:
            taskname_id.append(b[0])
            input_ids.append(b[1])
            max_length = max(max_length, b[2])
            input_length.append(b[2])
            answer_text_ids.append(b[3])

        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [self.pad_token_id] * (max_length - input_length[i])

        return taskname_id, input_ids, input_length, answer_text_ids


    def get_all_examples(self, tokens_to_generate):
        """
        Used for loading inference data.
        """
        task_id_nums, input_ids, answer_starts, answer_ids = zip(*self.examples)
        input_lengths = torch.cuda.LongTensor([len(inputs) for inputs in input_ids])
        task_id_nums = torch.cuda.LongTensor(task_id_nums)
        batch_max = input_lengths.max().item()
        batch_max += tokens_to_generate

        input_ids, _ = self.pad_batch_and_build_loss_mask(input_ids, batch_max, answer_starts)
        input_ids = input_ids.cuda()
        input_ids = torch.cuda.LongTensor(input_ids)

        return task_id_nums, (input_ids, input_lengths)

class IdentityTestPromptProcess():
    def __init__(
        self,
        tokenizer,
        virtual_prompt_source: VirtualPromptSource,
        task_templates: dict,
        pseudo_tokens,
        pad_token_id: str,
        max_seq_length: int,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = False,
        for_train: bool = False,
    ):
        self.tokenizer = tokenizer
        self.virtual_prompt_source = virtual_prompt_source
        self.task_templates = task_templates
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_token_ids = set(self.tokenizer.tokens_to_ids(self.pseudo_tokens))
        self.pad_token_id = pad_token_id
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.for_train = for_train
        self.taskname_ids = []
        self.input_ids = []
        self.input_lengths = []

    def encode(self, input_seqs, task_name):
        for seq in input_seqs:
          taskname = task_name
          prompt_template = self.task_templates[taskname]["prompt_template"]
          prompt_template_fields = self.task_templates[taskname]["prompt_template_fields"]
          total_virtual_tokens = self.task_templates[taskname]["total_virtual_tokens"]
          virtual_token_splits = self.task_templates[taskname]["virtual_token_splits"]
          truncation_field = self.task_templates[taskname]['truncate_field']
          answer_only_loss = self.task_templates[taskname]["answer_only_loss"]
          answer_field = self.task_templates[taskname]["answer_field"]

          input_example = prompt_template

          # Format the input example according to the template
          input_example = self._insert_text_in_template(input_example, prompt_template_fields, seq)
          input_example = self._insert_virtual_token_placeholders(input_example, virtual_token_splits)
          input_example = input_example.rstrip()
          input_ids = self.tokenizer.text_to_ids(input_example)

          # Add BOS/EOS if desired, adds EOS by default
          if self.add_bos:
              input_ids = [self.tokenizer.bos_id] + input_ids
          if self.add_eos:
              input_ids = input_ids + [self.tokenizer.eos_id]

          # Try to truncate input text to fit into the max sequence length
          if len(input_ids) > self.max_seq_length:
              input_ids = self._truncate_input(truncation_field, input_ids, taskname, seq)

          # Skip example if the final length doesn't fit length requirements even after truncation
          if self.min_seq_length <= len(input_ids) <= self.max_seq_length:
              if self.virtual_prompt_source == VirtualPromptSource.PROMPT_ENCODER:
                  taskname_id = self.tokenizer.text_to_ids(taskname)

              elif self.virtual_prompt_source == VirtualPromptSource.PROMPT_TABLE:
                  taskname_id = self.task_templates[taskname]["task_id_num"]

              self.taskname_ids.append(taskname_id)
              self.input_ids.append(input_ids)
              self.input_lengths.append(len(input_ids))

        return self.taskname_ids, self.input_ids, self.input_lengths

    def _input_sanity_checks(
        self,
        total_virtual_tokens,
        virtual_token_splits,
        prompt_template,
        prompt_template_fields,
        truncation_field,
        answer_only_loss,
        answer_field,
        doc,
    ):
        # Sanity check amount of virtual token
        assert total_virtual_tokens > 0, "There should be at least one virtual prompt token"
        assert (
            total_virtual_tokens < self.max_seq_length
        ), "virtual prompt tokens should not exceed max sequence length"

        # Make sure virtual token splits add up to the total number of virtual tokens
        assert (
            sum(virtual_token_splits) == total_virtual_tokens
        ), "Sum of prompt token split values must equal total number of prompt tokens"

        # Make sure number of virtual prompt locations match the number of virtual prompt splits
        assert prompt_template.count('<|VIRTUAL_PROMPT_') == len(
            virtual_token_splits
        ), "The number of '<|VIRTUAL_PROMPT_n|>' markers and the number of prompt token splits must match"

        # Check if input example has fields not present in template
        keys_not_in_template = list(set(doc.keys()) - set(prompt_template_fields) - set(['taskname']))
        assert (
            len(keys_not_in_template) == 0
        ), f"Examples in your dataset contain the fields: {keys_not_in_template} that are not in the task template."

        # Check that answer field checks if answer_only_loss was set to True
        if answer_only_loss and self.for_train:
            assert answer_field is not None, "If answer_only_loss=True, an answer_field must be given"
            assert (
                answer_field in doc.keys()
            ), f"answer_only_loss=True but the given answer_field '{answer_field}' is not in data json"
            assert truncation_field != answer_field, "Answer field and truncation field should not match"

            answer_placeholder = "{" + answer_field + "}"
            answer_placeholder_len = len(answer_placeholder)
            placeholder_start = len(prompt_template) - answer_placeholder_len
            assert prompt_template[placeholder_start:] == answer_placeholder, "Answer field must be at prompt end"

    def _insert_text_in_template(self, input_example, prompt_template_fields, doc):
        """ Format the input example according to the template """
        for field in prompt_template_fields:
            if field in doc.keys():
                field_text = doc[field]
                input_example = input_example.replace('{' + field + '}', field_text)

            # If some fields from the template aren't present, e.g. {answer} during inference
            # just remove that field from the template, leaving the space blank
            else:
                input_example = input_example.replace('{' + field + '}', "")

        return input_example

    def _insert_virtual_token_placeholders(self, input_example, virtual_token_splits):
        """ Insert the correct number of pseudo tokens at the <|virtual_PROMPT_n|> markers """
        total_inserted_tokens = 0

        for idx in range(len(virtual_token_splits)):
            split_start = total_inserted_tokens
            split_end = total_inserted_tokens + virtual_token_splits[idx]
            pseudo_tokens_for_split = "".join(self.pseudo_tokens[split_start:split_end])
            input_example = input_example.replace(f'<|VIRTUAL_PROMPT_{idx}|>', pseudo_tokens_for_split)
            total_inserted_tokens = split_end

        return input_example

    def _truncate_input(self, truncation_field, input_ids, taskname, doc):
        """ Try to truncate input text to fit into the max sequence length """
        logging.info(
            f"Input greater than max sequence length. Attempting to truncate: '{truncation_field}' in task: '{taskname}'"
        )

        # Truncate the text ids in this part of input to try and fit max sequence length
        if truncation_field is not None and truncation_field in doc.keys():
            truncation_length = len(input_ids) - self.max_seq_length
            field_text = doc[truncation_field]
            field_text = self._add_leading_space(taskname, truncation_field, field_text)

            # Truncate field text
            field_text_ids = self.tokenizer.text_to_ids(field_text)
            truncated_text_ids = field_text_ids[: -min(truncation_length, len(field_text_ids))]

            # Replace original text ids with truncated text ids
            field_start, field_end = find_subsequence_location(input_ids, field_text_ids)
            input_ids = input_ids[:field_start] + truncated_text_ids + input_ids[field_end + 1 :]

        return input_ids

    def _add_leading_space(self, taskname, field_name, field_text):
        """ Add leading space to text if there is a space before it in the template """
        prompt_template = self.task_templates[taskname]["prompt_template"]
        field_text_start = prompt_template.find("{" + field_name + "}")
        if field_text_start != 0 and prompt_template[field_text_start - 1] == " ":
            field_text = " " + field_text

        return field_text

def get_prompt_dataset(dataset_paths, tokenizer):

    prompt_dataset = GPTPromptLearningDataset(
        datasets=dataset_paths,
        tokenizer=tokenizer.tokenizer,
        virtual_prompt_source=tokenizer.virtual_prompt_source,
        task_templates=tokenizer.task_templates,
        pseudo_tokens=tokenizer.pseudo_tokens,
        pad_token_id=tokenizer.pad_token_id,
        max_seq_length=2048,
        min_seq_length=1,
        add_bos=False,
        add_eos=True,
        for_train=True,
    )
    return prompt_dataset

def get_pseudo_tokens(num_virtual_tokens):
    """
    Takes in an integer and returns a list of strings where each string
    is a numbered virtual token placeholder. If 
    num_virtual_tokens = 3, then this function returns:
    ["<prompt_0>", "<prompt_1>", "<prompt_2>"]
    Args:
        num_virtual_tokens: (int) Number of virtual token strings you want to make
    returns a list of string. 
    """
    pseudo_tokens = [
        VirtualPromptPlaceholderToken.BASE.value + str(i) + VirtualPromptPlaceholderToken.END.value
        for i in range(num_virtual_tokens)
    ]

    return pseudo_tokens


class GPTPromptLearningTokenizer:
    def __init__(self, model_path, prompt_path):
        model_cfg_path = model_path + "/model_config.yaml"
        prompt_cfg_path = prompt_path + "/model_config.yaml"
        self.model_cfg = OmegaConf.load(model_cfg_path)
        self.cfg = OmegaConf.load(prompt_cfg_path)
        self.existing_tasks = list(self.cfg.get('existing_tasks', []))
        self.new_tasks = list(self.cfg.get('new_tasks', []))

        # Load templates for assigning virtual prompt token positions
        self.load_task_templates(self.cfg.task_templates)

        # original gpt2 tokenizer
        vocab_file = model_path + "/" + self.model_cfg.vocab_file.split(":")[1]
        merges_file = model_path + "/" + self.model_cfg.merges_file.split(":")[1]

        # self.tokenizer = encoder.get_encoder(vocab_file, merge_file)
        model_name = "GPT2BPETokenizer"
        library = "megatron"
        self.tokenizer = get_nmt_tokenizer(library='megatron', model_name=model_name, vocab_file = vocab_file, merges_file = merges_file)
        
        # Prepare pseudo token ids for virtual/virtual prompt tokens
        self.pseudo_tokens = get_pseudo_tokens(self.max_virtual_tokens)
        self.tokenizer.add_special_tokens({'additional_special_tokens': self.pseudo_tokens})
        self.pseudo_token_ids = self.tokenizer.tokens_to_ids(self.pseudo_tokens)
        self.pseudo_token_ids_start = self.pseudo_token_ids[0]
        self.pad_token_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else self.tokenizer.unk_id

        self.virtual_prompt_style = VirtualPromptStyle(self.cfg.virtual_prompt_style)

        # Prompt tuning stores virtual prompts in the prompt table and tunes their weight directly
        if self.virtual_prompt_style in [VirtualPromptStyle.PROMPT_TUNING, VirtualPromptStyle.INFERENCE]:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_TABLE

        # P-Tuning uses an LSTM Encoder to produce virtual token embeddings
        elif self.virtual_prompt_style == VirtualPromptStyle.P_TUNING:
            self.virtual_prompt_source = VirtualPromptSource.PROMPT_ENCODER
        else:
            raise ValueError(
                f"\nvirtual prompt style '{cfg.virtual_prompt_style}' not recognized, please use one of 'prompt-tuning' or 'p-tuning'"
          )

    def load_task_templates(self, task_templates):
        """
        Takes in the task template portion of the config and turns  
        it into a table where each task's prompt template and 
        the number of virtual tokens to insert in a given part of 
        the prompt template are specified. 
        """
        self.task_templates = {}
        self.task_id_num_to_name = {}
        self.task_name_to_id_num = {}
        self.max_virtual_tokens = 0

        task_id_num = 0
        for task in task_templates:
            self.task_templates[task.taskname] = {
                "prompt_template": task.prompt_template,
                "prompt_template_fields": re.findall("\{(.*?)\}", task.prompt_template),
                "answer_only_loss": task.get("answer_only_loss", False),
                "answer_field": task.get("answer_field", None),
                "truncate_field": task.truncate_field,
                "total_virtual_tokens": task.total_virtual_tokens,
                "virtual_token_splits": task.virtual_token_splits,
                "task_id_num": task_id_num,
            }

            self.max_virtual_tokens = max(self.max_virtual_tokens, task.total_virtual_tokens)
            self.task_id_num_to_name[task_id_num] = task.taskname
            self.task_name_to_id_num[task.taskname] = task_id_num
            task_id_num += 1

        # Check that all new tasks have the same total num virtual tokens
        # Num virtual tokens for new tasks don't need to match num used for previously tuned tasks
        if self.new_tasks:
            new_task_name = self.new_tasks[0]
            self.total_new_task_virtual_tokens = self.task_templates[new_task_name]["total_virtual_tokens"]

            assert all(
                self.task_templates[taskname]["total_virtual_tokens"] == self.total_new_task_virtual_tokens
                for taskname in self.new_tasks
            ), "Total virtual tokens for each task tuned simultaneously must match. If you want to use a different number of virtual tokens for different tasks, tune them separately."

# prompt_tokenizer = GPTPomptLearningTokenizer('/workspace/nemo_prompt_learning2.0/models/multitask_p_tuned_gpt_125M/model_config.yaml')
