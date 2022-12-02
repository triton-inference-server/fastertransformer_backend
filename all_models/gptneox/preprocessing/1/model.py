# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from word_list import to_word_list_format, HFTokenizer


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
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

        # Parse model output configs and convert Triton types to numpy types
        input_names = ["INPUT_ID", "REQUEST_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS"]
        for input_name in input_names:
          setattr(self,
              input_name.lower() + "_dtype",
              pb_utils.triton_string_to_numpy(pb_utils.get_output_config_by_name(
                model_config, input_name)['data_type'])
          )

        cur_folder = Path(__file__).parent

        if self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf_t5":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["parameters"]["tokenizer_path"]["string_value"])
            self.start_id = self.tokenizer.eos_token_id
            self.end_id = self.tokenizer.bos_token_id
        elif self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf":
            self.tokenizer = HFTokenizer(str(self.model_config["parameters"]["tokenizer_path"]["string_value"]))
            self.start_id = 0
            self.end_id = 2
        else:
            assert False, f"{self.model_config['parameters']['tokenizer_type']['string_value']} is invalid"

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

            bad_words_dict = pb_utils.get_input_tensor_by_name(request, 'BAD_WORDS_DICT').as_numpy()
            stop_words_dict = pb_utils.get_input_tensor_by_name(request, 'STOP_WORDS_DICT').as_numpy()

            # Preprocessing input data.
            input_id, request_input_len = self._create_request(query)
            bad_words = to_word_list_format(bad_words_dict)
            stop_words = to_word_list_format(stop_words_dict)

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
            bad_words_ids_tensor = pb_utils.Tensor(
                'BAD_WORDS_IDS',
                bad_words)
            stop_words_ids_tensor = pb_utils.Tensor(
                'STOP_WORDS_IDS',
                stop_words)


            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor,
                bad_words_ids_tensor,
                stop_words_ids_tensor,
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


    def _create_request(self, query):
        """
            query : batch string (2D numpy array)
        """
        if self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf_t5":
            start_ids = [torch.IntTensor(self.tokenizer.encode(s[0].decode(),  add_special_tokens=False)) for s in query]
        elif self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf":
            start_ids = [torch.IntTensor(self.tokenizer.tokenize(s[0].decode())) for s in query]
        start_lengths = torch.IntTensor([[len(ids)] for ids in start_ids])

        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=self.end_id)

        return start_ids, start_lengths
