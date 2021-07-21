#!/usr/bin/bash

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

MODEL_FILENAME=gpt3_89B
BATCH_SIZE=16
INPUT_LEN=512
OUTPUT_LEN=16
SIZE_PER_HEAD=128
HEAD_NUM=96
MODEL_NAME="gpt3_89B"
VOCAB_SIZE=51200
NUM_DECODER_LAYERS=48
NUM_RUNS=1
SERVER_TIMEOUT=420
LAYER_PARA_SIZE=1 # This script only support single node, so keep this to be 1
source $WORKSPACE/common/util.sh

set +x


# TODO: Add TP para, now it only supports TP=8
function usage
{
    echo "usage: ./benchmark.sh [[-m model_filename ] | [-b batch_size] | [-h] | [-c compile] | [-i input_len] | [-o output_len] | [-l layer_para_size] | [-n num_decoder_layers]"
    echo "-m | --model_filename choose which model to run (gpt_89B, gpt_175B)"
    echo "-b | --batch_size"
    echo "-c | --compile recompile triton backend"
    echo "-i | --input_len"
    echo "-o | --output_len"
    echo "-v | --vocab_size"
    echo "-d | --num_decoder_layers"
    echo "-n | --num runs"
    echo "-h | --help  This message"
}


while [ "$1" != "" ]; do
    case $1 in
        -m | --model_filename)      shift
				    MODEL_FILENAME=$1
				    ;;
        -b | --batch_size)          shift
				    BATCH_SIZE=$1
				    ;;
        -c | --compile)             shift
				    shift
				    COMPILE=true
				    ;;
        -i | --input_len)           shift
				    INPUT_LEN=$1
				    ;;
        -o | --output_len)          shift
				    OUTPUT_LEN=$1
				    ;;
        -v | --vocab_size)          shift
				    VOCAB_SIZE=$1
				    ;;
        -d | --num_decoder_layers)  shift
				    NUM_DECODER_LAYERS=$1
				    ;;
	      -n | --num_runs)            shift
				    NUM_RUNS=$1
				    ;;
        -h | --help )               shift
				    usage
            exit 1
				    ;;
        * )                         usage
            exit 1
    esac
    shift
done

MAX_SEQ_LEN=$(( $INPUT_LEN + $OUTPUT_LEN ))

if [ "$COMPILE" = true ] ; then
    # Build
    if [[ -f $WORKSPACE/fastertransformer_backend/build/CMakeCache.txt ]]; then
        rm $WORKSPACE/fastertransformer_backend/build/CMakeCache.txt
    fi
    
    (cd $WORKSPACE/fastertransformer_backend/build/ && \
        cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DSM=80 .. && \
    make -j12)
    
    cp $WORKSPACE/fastertransformer_backend/build/libtriton_fastertransformer.so \
    $WORKSPACE/fastertransformer_backend/build/lib/libtransformer-shared.so \
    /opt/tritonserver/backends/fastertransformer
fi



RET=0
rm -rf *.log

SERVER=/opt/tritonserver/bin/tritonserver
MODEL_PATH=$WORKSPACE/fastertransformer_backend/all_models
SERVER_ARGS="--model-repositor=$MODEL_PATH"
SERVER_LOG="./inference_server.log"

#update config.pbtxt
(cd $MODEL_PATH/fastertransformer && \
    echo '
name: "fastertransformer"
backend: "fastertransformer"
default_model_filename: "'"${MODEL_FILENAME}"'"
max_batch_size: '"${BATCH_SIZE}"'
input [
  {
    name: "INPUT_ID"
    data_type: TYPE_UINT32
    dims: [ -1 ]
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
    dims: [ -1 ]
  }
]
instance_group [
  {
    count: 1
    kind : KIND_CPU
  }
]

parameters {
  key: "candidate_num"
  value: {
    string_value: "0"
  }
}
parameters {
  key: "probability_threshold"
  value: {
    string_value: "0.9"
  }
}
parameters {
  key: "tensor_para_size"
  value: {
    string_value: "8"
  }
}
parameters {
  key: "layer_para_size"
  value: {
    string_value: "'"${LAYER_PARA_SIZE}"'"
  }
}
parameters {
  key: "layer_para_batch_size"
  value: {
    string_value: "'"${BATCH_SIZE}"'"
  }
}
parameters {
  key: "max_seq_len"
  value: {
    string_value: "'"${MAX_SEQ_LEN}"'"
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
    string_value: "'"${HEAD_NUM}"'"
  }
}
parameters {
  key: "size_per_head"
  value: {
    string_value: "'"${SIZE_PER_HEAD}"'"
  }
}
parameters {
  key: "vocab_size"
  value: {
    string_value: "'"${VOCAB_SIZE}"'"
  }
}
parameters {
  key: "decoder_layers"
  value: {
    string_value: "'"${NUM_DECODER_LAYERS}"'"
  }
}
parameters {
  key: "model_name"
  value: {
    string_value: "'"${MODEL_NAME}"'"
  }
}
parameters {
  key: "batch_size"
  value: {
    string_value: "'"${BATCH_SIZE}"'"
  }
}
parameters {
  key: "is_fuse_QKV"
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
' > config.pbtxt)



# start server
echo "Starting server..."
mpirun -n 1 --allow-run-as-root $SERVER $SERVER_ARGS 2>&1 | tee $SERVER_LOG &

SERVER_PID=$!
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start server\n***"
    cat $SERVER_LOG
    exit 1
fi
echo $SERVER_PID

wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT


CLIENT_PY=$WORKSPACE/fastertransformer_backend/tools/identity_test.py
CLIENT_LOG="./client.log"

rm -rf client.log err.log
rm -rf triton_out

for PROTOCOL in http; do
    set +e
    python $CLIENT_PY -i $PROTOCOL -b $BATCH_SIZE -s $INPUT_LEN -o $OUTPUT_LEN -n $NUM_RUNS -v 2> err.log > $CLIENT_LOG
    if [ $? -ne 0 ]; then
        RET=1
    fi
    set -e
done

# latency will be logged to last line
tail -n 2 $CLIENT_LOG
# kill server
ps -ef | grep mpirun | grep triton | awk '{print $2}' | while read p; do kill -9 $p ; done

echo "model_name = $MODEL_NAME, batch_size = $BATCH_SIZE, input_len = $INPUT_LEN, output_len = $OUTPUT_LEN, num_decoder_layers = $NUM_DECODER_LAYERS latency = $(tail -n 1 $CLIENT_LOG | grep -Eo '[+-]?[0-9]+([.][0-9]+)?') ms"

exit $RET
