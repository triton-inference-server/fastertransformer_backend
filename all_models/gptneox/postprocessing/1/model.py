# -*- coding: utf-8 -*-
import json
import numpy as np
import triton_python_backend_utils as pb_utils

from pathlib import Path
from tokenizers import Tokenizer
from typing import List, Union

class HFTokenizer:
    def __init__(self, vocab_file):
        self.tokenizer = Tokenizer.from_file(vocab_file)

    def tokenize(self, text: str):
        return self.tokenizer.encode(text).ids

    def tokenize_batch(self, text_batch: Union[List[str], str]):
        return self.tokenizer.encode_batch(text_batch)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)


TOKENIZER_FILE = "20B_tokenizer.json"

MAX_BATCH_SIZE = 8

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

        # Parse model output configs 
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT")

        # Convert Triton types to numpy types
        self.output_dtype= pb_utils.triton_string_to_numpy(
            output_config['data_type'])

        if self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf_t5":
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config["parameters"]["tokenizer_path"]["string_value"])
        elif self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf":
            self.tokenizer = HFTokenizer(str(self.model_config["parameters"]["tokenizer_path"]["string_value"]))
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
            tokens_batch = pb_utils.get_input_tensor_by_name(request, 'TOKENS_BATCH').as_numpy()
            sequence_length = pb_utils.get_input_tensor_by_name(request, 'sequence_length').as_numpy()

            # Reshape Input 
            # tokens_batch = tokens_batch.reshape([-1, tokens_batch.shape[0]])
            # tokens_batch = tokens_batch.T

            # Postprocessing output data.
            outputs = self._postprocessing(tokens_batch, sequence_length)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor(
                'OUTPUT',
                np.array(outputs).astype(self.output_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                output_tensor])
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


    def _postprocessing(self, tokens_batch, sequence_length):
        cur_folder = Path(__file__).parent

        outputs = []
        for beam_tokens, beam_len in zip(tokens_batch, sequence_length):
            for tokens, len in zip(beam_tokens, beam_len):
                if self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf_t5":
                    output = self.tokenizer.decode(tokens[:len])
                elif self.model_config["parameters"]["tokenizer_type"]["string_value"] == "hf":
                    output = self.tokenizer.detokenize(tokens)
                outputs.append(output.encode('utf8'))
        return outputs 
