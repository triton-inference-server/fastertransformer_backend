import json
import numpy as np
import triton_python_backend_utils as pb_utils

from pathlib import Path
from transformers import AutoTokenizer


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
        self.config = json.loads(args["model_config"])
        self.current_dir = Path(__file__).parent
        model_max_length = int(self.config["parameters"]["model_max_length"]["string_value"])
        reference_model  = self.config["parameters"]["reference_model"]["string_value"]
        self.tokenizer = AutoTokenizer.from_pretrained(reference_model,
                                                       model_max_length=model_max_length,
                                                       cache_dir=self.current_dir / ".cache")
        print('Initialized...')

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

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

        # Every Python backend must iterate through list of requests and create
        # an instance of pb_utils.InferenceResponse class for each of them. You
        # should avoid storing any of the input Tensors in the class attributes
        # as they will be overridden in subsequent inference requests. You can
        # make a copy of the underlying NumPy array and store it if it is
        # required.
        for request in requests:
            # Perform inference on the request and append it to responses list...
            query = pb_utils.get_input_tensor_by_name(request, "query")
            query = [sentence[0] for sentence in query.as_numpy().astype(str)]

            ret = self.tokenizer(query, padding=True, return_tensors="np")

            ids = pb_utils.Tensor("input_ids", ret["input_ids"].astype(np.uint32))

            len = np.array([[size] for size in np.sum(ret["attention_mask"], axis=1)])
            len = pb_utils.Tensor("request_input_len", len.astype(np.uint32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[ids, len]))

        # You must return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses
