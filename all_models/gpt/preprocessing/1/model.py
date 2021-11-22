# -*- coding: utf-8 -*-
#  Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
#  Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.

#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import triton_python_backend_utils as pb_utils

import utils.gpt_token_encoder as encoder


# GPT3 Related variables
# Reference : https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/gpt_sample.py
MERGES_FILE = "gpt2-merges.txt"
VOCAB_FILE = "gpt2-vocab.json"

START_ID = 50256
END_ID = 50256
MAX_BATCH_SIZE = 8

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])

        # Parse model output configs 
        input_id_config = pb_utils.get_output_config_by_name(
            model_config, "INPUT_ID")
        request_input_len_config = pb_utils.get_output_config_by_name(
            model_config, "REQUEST_INPUT_LEN")

        # Convert Triton types to numpy types
        self.input_id_dtype = pb_utils.triton_string_to_numpy(
            input_id_config['data_type'])
        self.request_input_len_dtype = pb_utils.triton_string_to_numpy(
            request_input_len_config['data_type'])

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors 
            query = pb_utils.get_input_tensor_by_name(request, 'QUERY').as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(request, 'REQUEST_OUTPUT_LEN').as_numpy()

            # Preprocessing input data.
            input_id, request_input_len = self._preprocessing(query)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID',
                np.array(input_id).astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                np.array(request_input_len).astype(self.request_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN',
                request_output_len)


            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor,
                request_input_len_tensor,
                request_output_len_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses


    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')


    def _preprocessing(self, query) :
        """
            query : batch string (2D numpy array)
        """
        cur_folder = Path(__file__).parent
        enc = encoder.get_encoder(str(cur_folder/VOCAB_FILE), str(cur_folder/MERGES_FILE))

        # Inputs
        start_ids = [torch.IntTensor(enc.encode(s[0].decode())) for s in query]

        start_lengths = [[len(ids)] for ids in start_ids]
        input_len = min(start_lengths)

        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=END_ID)
        start_ids = start_ids.reshape([start_ids.shape[0], -1, start_ids.shape[-1]])
        start_lengths = torch.IntTensor(start_lengths)
        #attn_mask = torch.ones((batch_size, input_len, input_len)).tril()
        return start_ids, start_lengths
