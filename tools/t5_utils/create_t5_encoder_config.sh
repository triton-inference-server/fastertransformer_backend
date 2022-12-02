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
elif [ "$DATA_TYPE" = "bf16" ]; then
  TRITON_TYPE=TYPE_BF16
elif [ "$DATA_TYPE" = "fp32" ]; then
  TRITON_TYPE=TYPE_FP32
else
  echo "[ERROR] ${DATA_TYPE} is invalid."
    exit
fi

echo "
name: \"fastertransformer\"
backend: \"fastertransformer\"
default_model_filename: \"t5-encoder\"
max_batch_size: 1024
input [
  {
    name: \"input_ids\"
    data_type: TYPE_UINT32
    dims: [ -1 ]
  },
  {
    name: \"sequence_length\"
    data_type: TYPE_UINT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
  },
  {
    name: \"is_return_attentions\"
    data_type: TYPE_BOOL
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  },
  {
    name: \"ia3_tasks\"
    data_type: TYPE_INT32
    dims: [ 1 ]
    reshape: { shape: [ ] }
    optional: true
  }
]
output [
  {
    name: \"output_hidden_state\"
    data_type: ${TRITON_TYPE} # return type will be set automatically based on model's data_type
    dims: [ -1, -1 ]
  },
  {
    name: \"output_attentions\"
    data_type: ${TRITON_TYPE} # return type will be set automatically based on model's data_type
    dims: [-1, -1, -1, -1 ] # [num_layers, num_head, sequence_length, sequence_length]
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
    string_value: \"T5-Encoder\"
  }
}
parameters {
  key: \"model_checkpoint_path\"
  value: {
    string_value: \"${MODEL_PATH}\"
  }
}
" > config.pbtxt