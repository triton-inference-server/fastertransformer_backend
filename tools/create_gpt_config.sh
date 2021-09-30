echo '
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

name: "fastertransformer"
backend: "fastertransformer"
default_model_filename: "gpt3_345M"
max_batch_size: 128
input [
  {
    name: "INPUT_ID"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  },
  {
    name: "REQUEST_INPUT_LEN"
    data_type: TYPE_UINT32
    dims: [ 1 ]
  },
  {
    name: "REQUEST_OUTPUT_LEN"
    data_type: TYPE_UINT32
    dims: [ 1 ]
  }
]
output [
  {
    name: "OUTPUT0"
    data_type: TYPE_UINT32
    dims: [ -1, -1 ]
  }
]
instance_group [
  {
    count: 1
    kind : KIND_CPU
  }
]
parameters {
  key: "top_k"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "top_p"
  value: {
    string_value: "0.0"
  }
}
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "pipeline_para_size"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "max_input_len"
  value: {
    string_value: "512"
  }
}
parameters {
  key: "max_seq_len"
  value: {
    string_value: "528"
  }
}
parameters {
  key: "is_half"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "head_num"
  value: {
    string_value: "16"
  }
}
parameters {
  key: "size_per_head"
  value: {
    string_value: "64"
  }
}
parameters {
  key: "inter_size"
  value: {
    string_value: "4096"
  }
}
parameters {
  key: "vocab_size"
  value: {
    string_value: "50304"
  }
}
parameters {
  key: "start_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "end_id"
  value: {
    string_value: "50256"
  }
}
parameters {
  key: "decoder_layers"
  value: {
    string_value: "24"
  }
}
parameters {
  key: "model_name"
  value: {
    string_value: "gpt3_345M"
  }
}
parameters {
  key: "beam_width"
  value: {
    string_value: "1"
  }
}
parameters {
  key: "temperature"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "repetition_penalty"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "len_penalty"
  value: {
    string_value: "1.0"
  }
}
parameters {
  key: "beam_search_diversity_rate"
  value: {
    string_value: "0.0"
  }
}
dynamic_batching {
  preferred_batch_size: [4, 8]
  max_queue_delay_microseconds: 200000
}
parameters {
  key: "model_type"
  value: {
    string_value: "GPT"
  }
}
' >> config.pbtxt