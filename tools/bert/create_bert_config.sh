MODEL_PATH=$1
TP=$2
PP=$3
DATA_TYPE=$4

if [ $MODEL_PATH ]; then
  :
else
	echo "MODEL_PATH IS NOT EXISTS"
    exit
fi

if [ $TP ]; then
  :
else
	echo "TP IS NOT EXISTS"
    exit
fi

if [ $PP ]; then
  :
else
	echo "PP IS NOT EXISTS"
    exit
fi

if [ $DATA_TYPE ]; then
  :
else
	echo "DATA_TYPE IS NOT EXISTS"
    exit
fi

if [ "$DATA_TYPE" = "fp16" ]; then
  TRITON_TYPE=TYPE_FP16
elif [ "$DATA_TYPE" = "fp32" ]; then
  TRITON_TYPE=TYPE_FP32
else
  echo "[ERROR] ${DATA_TYPE} is invalid."
    exit
fi

echo "
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS AND ANY
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

name: \"fastertransformer\"
backend: \"fastertransformer\"
default_model_filename: \"bert\"
max_batch_size: 1024
input [
  {
    name: \"input_hidden_state\"
    data_type: ${TRITON_TYPE}
    dims: [ -1, -1 ]
  },
  {
    name: \"sequence_lengths\"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  }
]
output [
  {
    name: \"output_hidden_state\"
    data_type: ${TRITON_TYPE}
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
  key: \"tensor_para_size\"
  value: {
    string_value: \"${TP}\"
  }
}
parameters {
  key: \"pipeline_para_size\"
  value: {
    string_value: \"${PP}\"
  }
}
parameters {
  key: \"data_type\"
  value: {
    string_value: \"${DATA_TYPE}\"
  }
}
parameters {
  key: \"enable_custom_all_reduce\"
  value: {
    string_value: \"0\"
  }
}
parameters {
  key: \"model_type\"
  value: {
    string_value: \"bert\"
  }
}
parameters {
  key: \"model_checkpoint_path\"
  value: {
    string_value: \"${MODEL_PATH}\"
  }
}
parameters {
  key: \"int8_mode\"
  value: {
    string_value: \"0\"
  }
}
parameters {
  key: \"is_sparse\"
  value: {
    string_value: \"0\"
  }
}
parameters {
  key: \"is_remove_padding\"
  value: {
    string_value: \"1\"
  }
}
" >> config.pbtxt