// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdint.h>

#include <exception>
#include <string>
#include <thread>
#include <vector>

#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma warning(push, 0)
#pragma warning(pop)
#pragma GCC diagnostic pop

// must include triton libraries first
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

// FT's libraries have dependency with triton's lib
#include "src/fastertransformer/triton_backend/bert/BertTritonModel.h"
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModel.h"
#include "src/fastertransformer/triton_backend/gptj/GptJTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/gptneox/GptNeoXTritonModel.h"
#include "src/fastertransformer/triton_backend/gptneox/GptNeoXTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModel.h"
#include "src/fastertransformer/triton_backend/multi_gpu_gpt/ParallelGptTritonModelInstance.h"
#include "src/fastertransformer/triton_backend/t5/T5TritonModel.h"
#include "src/fastertransformer/triton_backend/t5/T5TritonModelInstance.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/mpi_utils.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;

namespace triton { namespace backend { namespace fastertransformer_backend {

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                 \
    TRITONSERVER_Error* raarie_err__ = (X);                            \
    if (raarie_err__ != nullptr) {                                     \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__); \
      return;                                                          \
    }                                                                  \
  } while (false)

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  TRITONSERVER_Error* LoadModel(
      const std::string& artifact_name, const int32_t node_id,
      const int32_t device_id, const int32_t device_id_start, const int32_t stream_id,
      std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>& nccl_params,
      std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comms,
      std::string* model_path,
      std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance);

  int GetGpuSize() { return gpu_size; };
  int GetWorldSize() { return world_size; };
  int GetParallelSize() { return tp_pp_size; };
  int GetInstanceId() { return current_model_instance_id++; };
  int GetInstanceGroupCount() { return instance_group_count; };
  bool SequenceBatchingEnabled() { return sequence_batching_enabled; };
  std::shared_ptr<AbstractTransformerModel> GetFtModel() { return ft_model; };

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  int current_model_instance_id = 0;
  bool sequence_batching_enabled = false;
  int instance_group_count = 1;
  std::shared_ptr<AbstractTransformerModel> ft_model;
  int node_id, gpu_size, world_size, tp_pp_size;
  std::vector<cudaStream_t> streams_;
};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(
      triton_model, &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR((*state)->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    (*state)->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message* message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }

  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model, true)
{
  node_id = ft::mpi::getCommWorldRank();
  int num_nodes = ft::mpi::getCommWorldSize();

  triton::common::TritonJson::WriteBuffer buffer;
  ModelConfig().PrettyWrite(&buffer);
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value param;
  model_config_.MemberAsObject("parameters", &param);
  auto param_get = [&](const char* field) {
    common::TritonJson::Value key;
    std::string value;
    param.MemberAsObject(field, &key);
    key.MemberAsString("string_value", &value);
    return value;
  };
  auto param_get_int = [&](const char* field) {
    int ret = 0;
    try {
      ret = std::stoi(param_get(field));
    }
    catch (std::invalid_argument& ia) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Invalid configuration argument '") + field +
           "': " + ia.what())
              .c_str());
    }
    return ret;
  };
  auto param_get_float = [&](const char* field) {
    float ret = 0.0;
    try {
      ret = std::stof(param_get(field));
    }
    catch (std::invalid_argument& ia) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Invalid configuration argument '") + field +
           "': " + ia.what())
              .c_str());
    }
    return ret;
  };

  // instance groups
  triton::common::TritonJson::Value instance_group, instance_obj,
    instance_group_count_val, instance_group_kind;
  if (!ModelConfig().Find("instance_group", &instance_group) ||
    instance_group.ArraySize() > 1) {
    THROW_IF_BACKEND_MODEL_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
      "Only supports one instance group !"));
  }
  instance_group.IndexAsObject(0, &instance_obj);
  instance_obj.Find("count", &instance_group_count_val);
  instance_obj.Find("kind", &instance_group_kind);
  std::string instance_group_kind_str;
  int64_t instance_group_count_int64 = 1;
  instance_group_kind.AsString(&instance_group_kind_str);
  instance_group_count_val.AsInt(&instance_group_count_int64);
  instance_group_count = (int) instance_group_count_int64;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, ("Instance group type: " + instance_group_kind_str +
    " count: " + std::to_string(instance_group_count_int64)).c_str());
  if (instance_group_kind_str != "KIND_CPU") {
    THROW_IF_BACKEND_MODEL_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
      "Instance Group: only KIND_CPU supports!"));
  }

  // instance group validation
  bool multi_node_enabled = num_nodes > 1;
  tp_pp_size = param_get_int("tensor_para_size") * param_get_int("pipeline_para_size");
  gpu_size = ft::getDeviceCount();
  world_size = gpu_size * num_nodes;
  int model_instance_size = num_nodes > 1 ? gpu_size : tp_pp_size;
  bool multi_model_instance_valid = (multi_node_enabled && tp_pp_size == world_size &&
    instance_group_count == 1) || (!multi_node_enabled && gpu_size % tp_pp_size == 0 &&
    model_instance_size * instance_group_count >=  gpu_size);
  if (!multi_model_instance_valid) {
    THROW_IF_BACKEND_MODEL_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
      "1. Number of visible GPUs must be evenly divisble by TP * PP \n"
      "2. Number of visible GPUs must be <= instance count * TP * PP \n"
      "3. Multi-Node Inference only support one model instance \n"));
  }

  int64_t max_batch_size = 0;
  model_config_.MemberAsInt("max_batch_size", &max_batch_size);

  // sequence batching
  triton::common::TritonJson::Value squence_batching;
  sequence_batching_enabled = ModelConfig().Find("sequence_batching", &squence_batching);
  std::string sequence_batching_log = sequence_batching_enabled ? "enabled" : "disabled";
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Sequence Batching: ") + sequence_batching_log).c_str());
  if (sequence_batching_enabled && max_batch_size != 1) {
    THROW_IF_BACKEND_MODEL_ERROR(TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
      "Sequence Batching for interactive text generation: only supports max batch size = 1 currently !"));
  }

  std::string model_filename;
  model_config_.MemberAsString("default_model_filename", &model_filename);

  model_filename =
      model_filename == ""
          ? std::to_string(param_get_int("tensor_para_size")) + "-gpu"
          : model_filename;

  std::string model_dir =
      param_get("model_checkpoint_path") == ""
          ? JoinPath(
                {RepositoryPath(), std::to_string(Version()), model_filename})
          : param_get("model_checkpoint_path");

  std::string model_type =
      param_get("model_type") == "" ? "GPT" : param_get("model_type");

  std::string data_type;
  if (model_type == "GPT" || model_type == "GPT-J" ||
      model_type == "GPT-NeoX" || model_type == "T5" || model_type == "bert") {
    data_type = param_get("data_type");
  }

  if (model_type == "GPT") {
    if (data_type == "fp16") {
      ft_model.reset(new ParallelGptTritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode")));
#ifdef ENABLE_BF16
    } else if (data_type == "bf16") {
      ft_model.reset(new ParallelGptTritonModel<__nv_bfloat16>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode")));
#endif
    } else if (data_type == "fp32") {
      ft_model.reset(new ParallelGptTritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode")));
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Invalid configuration argument 'data_type': ") +
           data_type)
              .c_str());
    }
  } else if (model_type == "GPT-J") {
    if (data_type == "fp16") {
      ft_model.reset(new GptJTritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
#ifdef ENABLE_BF16
    } else if (data_type == "bf16") {
      ft_model.reset(new GptJTritonModel<__nv_bfloat16>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
#endif
    } else if (data_type == "fp32") {
      ft_model.reset(new GptJTritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Invalid configuration argument 'data_type': ") +
           data_type)
              .c_str());
    }
  } else if (model_type == "GPT-NeoX") {
    if (data_type == "fp16") {
      ft_model.reset(new GptNeoXTritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
    } else {
      ft_model.reset(new GptNeoXTritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
    }
  } else if (model_type == "GPT-NeoX") {
    if (data_type == "fp16") {
      ft_model.reset(new GptNeoXTritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
    } else {
      ft_model.reset(new GptNeoXTritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir));
    }
  } else if (model_type == "T5") {
    if (data_type == "fp16") {
      ft_model.reset(new T5TritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir, 0));
#ifdef ENABLE_BF16
    } else if (data_type == "bf16") {
      ft_model.reset(new T5TritonModel<__nv_bfloat16>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir, 0));
#endif
    } else if (data_type == "fp32") {
      ft_model.reset(new T5TritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir, 0));
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("Invalid configuration argument 'data_type': ") +
           data_type)
              .c_str());
    }
  } else if (model_type == "bert") {
    if (data_type == "fp16") {
      ft_model.reset(new BertTritonModel<half>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode"),
          static_cast<bool>(param_get_int("is_sparse")),
          static_cast<bool>(param_get_int("is_remove_padding"))));
    } else if (data_type == "fp32") {
      ft_model.reset(new BertTritonModel<float>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode"),
          static_cast<bool>(param_get_int("is_sparse")),
          static_cast<bool>(param_get_int("is_remove_padding"))));
#ifdef ENABLE_BF16
    } else if (data_type == "bf16") {
      ft_model.reset(new BertTritonModel<__nv_bfloat16>(
          param_get_int("tensor_para_size"),
          param_get_int("pipeline_para_size"),
          param_get_int("enable_custom_all_reduce"), model_dir,
          param_get_int("int8_mode"),
          static_cast<bool>(param_get_int("is_sparse")),
          static_cast<bool>(param_get_int("is_remove_padding"))));
#endif
    }
  }

  int total_weight_gpu_size = (instance_group_count * model_instance_size) >= gpu_size ?
    gpu_size : (instance_group_count * model_instance_size);
  streams_.resize(instance_group_count * model_instance_size);

  /* create shared weights
  assume 8 gpus, 8 model instances, Tensor Para Size 2
  then we will distribute model instances to [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 1], [2, 3], [4, 5], [6, 7] GPUs;
  two instance instances on GPUs [0, 1] will share the same weights
  */
  std::vector<std::thread> threads;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Before Loading Weights:")).c_str());
  ft::print_mem_usage();
  for (int gid = 0; gid < total_weight_gpu_size; gid++) {
    int rank = node_id * gpu_size + gid % tp_pp_size;
    threads.push_back(std::thread(&AbstractTransformerModel::createSharedWeights,
      ft_model, gid, rank));
  }
  for (auto& t : threads) {
    t.join();
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("After Loading Weights:")).c_str());
  ft::print_mem_usage();
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name, const int32_t node_id,
    const int32_t device_id, const int32_t device_id_start, const int32_t stream_id,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>>&
        nccl_params_instance,
    std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comms,
    std::string* model_path,
    std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance)
{
  ft::check_cuda_error(cudaSetDevice(device_id));
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "gpt3-model";
  }

  if (!node_id && !device_id) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, (std::string("Before Loading Model:")).c_str());
  }
  ft::print_mem_usage();

  cudaStreamCreate(&streams_[stream_id]);
  const int rank = node_id * GetGpuSize() + device_id - device_id_start;
  ft::sync_check_cuda_error();
  auto model_instance = ft_model->createModelInstance(
      device_id, rank, streams_[stream_id], nccl_params_instance,
      custom_all_reduce_comms);
  ft_model_instance->reset(model_instance.release());

  if (!node_id && !device_id) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, (std::string("After Loading Model:")).c_str());
  }
  ft::print_mem_usage();

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since fastertransformer does
  // not store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for fastertransformer backend")
          .c_str());

  return nullptr;  // success
}

struct stream_callback_ctx_t {
  size_t total_batch_size;
  TRITONBACKEND_Request** requests;
  uint32_t request_count;
  std::vector<TRITONBACKEND_Response*>* responses;
  std::vector<TRITONBACKEND_ResponseFactory*>* factories;
  BackendModelInstance* model;
};

void
generate_response_placeholders(
    std::vector<TRITONBACKEND_Response*>* responses,
    std::vector<TRITONBACKEND_ResponseFactory*>* factories)
{
  TRITONSERVER_Error* err = nullptr;
  for (auto factory : *factories) {
    TRITONBACKEND_Response* response;
    err = TRITONBACKEND_ResponseNewFromFactory(&response, factory);
    if (err) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR, "Fail to create response from factory");
      TRITONSERVER_ErrorDelete(err);
    }
    responses->push_back(response);
  }
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  std::shared_ptr<std::unordered_map<std::string, Tensor>> Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      stream_callback_ctx_t* context, const uint32_t response_count,
      std::shared_ptr<std::unordered_map<std::string, Tensor>> input_tensors);

  void ReadOutputTensors(
      size_t total_batch_size,
      std::shared_ptr<std::unordered_map<std::string, Tensor>> output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  int GetModelInstanceCount() { return model_instance_count_; };
  int GetModelInstanceId() { return model_instance_id_; };

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);
  TRITONSERVER_Error* ValidateInputs();
  TRITONSERVER_Error* ValidateOutputs();

  void SetInputTensors(
      size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      BackendInputCollector* collector, std::vector<const char*>* input_names,
      std::shared_ptr<std::unordered_map<std::string, Tensor>>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);

  void BroadcastInputTensors(
    std::shared_ptr<std::unordered_map<std::string, Tensor>>* input_tensors);

  ModelState* model_state_;

  // model instance id
  int model_instance_count_ = 1;
  int model_instance_id_ = 0;
  int model_instance_gpu_size_ = 1;
  int model_instance_device_id_start_ = 0;

  // output tensor stream
  cudaStream_t output_stream_;

  // tensor parallel + pipeline parallel
  int gpu_size_ = 1;
  int world_size_ = 1;
  int tp_pp_size_ = 1;

  // Should we use the streaming API?
  bool is_decoupled_ = false;

  // The full path to the FT model file.
  std::string model_path_;

  std::vector<std::unique_ptr<AbstractTransformerModelInstance>>
      ft_model_instance_;

  // inter-node broadcast buffer
  std::vector<char*> bcast_buffers;

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;

  std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params_;

  // custom all reduce comms
  std::vector<std::shared_ptr<ft::AbstractCustomComm>> custom_all_reduce_comms_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }

  return nullptr;  // success
}

int
ThreadLoadModel(
    ModelState* model_state, const std::string& artifact_name,
    const int32_t node_id, const int32_t device_id, const int32_t device_id_start,
    const int32_t stream_id,
    std::pair<std::vector<ft::NcclParam>, std::vector<ft::NcclParam>> nccl_params,
    std::shared_ptr<ft::AbstractCustomComm> custom_all_reduce_comms,
    std::string* model_path,
    std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance)
{
  THROW_IF_BACKEND_INSTANCE_ERROR(model_state->LoadModel(
      artifact_name, node_id, device_id, device_id_start, stream_id,
      nccl_params, custom_all_reduce_comms, model_path, ft_model_instance));
  return 0;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  int node_id = ft::mpi::getCommWorldRank();
  int num_nodes = ft::mpi::getCommWorldSize();

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("Model name ") + ArtifactFilename()).c_str());

  triton::common::TritonJson::Value transaction_policy;
  is_decoupled_ = false;
  model_state_->ModelConfig().MemberAsObject(
      "model_transaction_policy", &transaction_policy);
  transaction_policy.MemberAsBool("decoupled", &is_decoupled_);

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("Use ") +
       (is_decoupled_ ? "DECOUPLED (streaming)" : "COUPLED (classic)") +
       " API.")
          .c_str());

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());

  // NOTE:  model instance params
  model_instance_id_ = model_state->GetInstanceId();
  model_instance_count_ = model_state->GetInstanceGroupCount();
  tp_pp_size_ = model_state->GetParallelSize();
  gpu_size_ = model_state->GetGpuSize();
  world_size_ = model_state->GetWorldSize();

  model_instance_gpu_size_ = num_nodes > 1 ? gpu_size_ : tp_pp_size_;
  ft_model_instance_.resize(model_instance_gpu_size_);
  std::vector<std::thread> threads;

  std::shared_ptr<AbstractTransformerModel> shared_ft_model =
      model_state->GetFtModel();

  // NOTE: CPU_KIND only, the backend fully controls how to distribute models to GPUs
  model_instance_device_id_start_ = (model_instance_id_ * model_instance_gpu_size_) % gpu_size_;
  // create output tensor stream
  ft::check_cuda_error(cudaSetDevice(model_instance_device_id_start_));
  ft::check_cuda_error(cudaStreamCreate(&output_stream_));
  // create nccl params
  nccl_params_ = shared_ft_model->createNcclParams(node_id, model_instance_device_id_start_, num_nodes > 1);

  shared_ft_model->createCustomComms(&custom_all_reduce_comms_, world_size_);
  std::string model_instance_gpu_ids = "[ ";
  for (int gid = model_instance_device_id_start_;
    gid < model_instance_device_id_start_ + model_instance_gpu_size_; gid++)
  {
    model_instance_gpu_ids += (std::to_string(gid) + " ");
    threads.push_back(std::thread(
        ThreadLoadModel, model_state, ArtifactFilename(), node_id, gid,
        model_instance_device_id_start_,
        model_instance_id_ * model_instance_gpu_size_ + gid,  nccl_params_,
        custom_all_reduce_comms_[gid - model_instance_device_id_start_],
        &model_path_, &ft_model_instance_[gid - model_instance_device_id_start_]));
  }
  model_instance_gpu_ids += "]";

  for (auto& t : threads) {
    t.join();
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Model instance is created on GPU ") + model_instance_gpu_ids).c_str());
}

ModelInstanceState::~ModelInstanceState(){
#ifdef TRITON_ENABLE_GPU
#endif  // TRITON_ENABLE_GPU
    for (auto bcast_buffer : bcast_buffers) {
      free(bcast_buffer);
    }
}

TRITONSERVER_Error* ModelInstanceState::ValidateInputs()
{
  triton::common::TritonJson::Value ios;
  std::string name, data_type;
  triton::common::TritonJson::Value jshape;
  model_state_->ModelConfig().MemberAsArray("input", &ios);

  for (size_t size = 0; size < ios.ArraySize(); size++) {
    triton::common::TritonJson::Value input;
    ios.IndexAsObject(size, &input);
    input.MemberAsString("name", &name);
    input.MemberAsString("data_type", &data_type);
    input.MemberAsArray("dims", &jshape);

    std::vector<int64_t> shape;
    for (size_t size = 0; size < jshape.ArraySize(); size++) {
      int64_t value = 0;
      jshape.IndexAsInt(size, &value);
      shape.push_back(value);
    }

    std::string str_shape = "[";
    for (uint i = 0; i < shape.size(); i++) {
      str_shape = str_shape + std::to_string(shape[i]);
      if (i != shape.size() - 1) {
        str_shape = str_shape + ", ";
      } else {
        str_shape = str_shape + "]";
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN, (std::string(
                                    "Get input name: " + name + ", type: " +
                                    data_type + ", shape: " + str_shape)
                                    .c_str()));
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::ValidateOutputs()
{
  triton::common::TritonJson::Value ios;
  RETURN_IF_ERROR(model_state_->ModelConfig().MemberAsArray("output", &ios));

  std::string name, data_type;
  triton::common::TritonJson::Value jshape;
  model_state_->ModelConfig().MemberAsArray("output", &ios);
  for (size_t size = 0; size < ios.ArraySize(); size++) {
    triton::common::TritonJson::Value input;
    ios.IndexAsObject(size, &input);
    input.MemberAsString("name", &name);
    input.MemberAsString("data_type", &data_type);
    input.MemberAsArray("dims", &jshape);

    std::vector<int64_t> shape;
    for (size_t size = 0; size < jshape.ArraySize(); size++) {
      int64_t value = 0;
      jshape.IndexAsInt(size, &value);
      shape.push_back(value);
    }

    std::string str_shape = "[";
    for (uint i = 0; i < shape.size(); i++) {
      str_shape = str_shape + std::to_string(shape[i]);
      if (i != shape.size() - 1) {
        str_shape = str_shape + ", ";
      } else {
        str_shape = str_shape + "]";
      }
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN, (std::string(
                                    "Get output name: " + name + ", type: " +
                                    data_type + ", shape: " + str_shape)
                                    .c_str()));
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  const int max_batch_size = model_state_->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  bool sequence_batching_enabled = model_state_->SequenceBatchingEnabled();
  size_t real_batch_dim = (int) sequence_batching_enabled;
  // only one batch slot per model instance when sequence_batching enabled
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to FasterTransformer backend for '" +
                  Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      int index = 0;
      while (true) {
        TRITONBACKEND_Input* input;
        TRITONSERVER_Error* err_0 =
            TRITONBACKEND_RequestInputByIndex(requests[i], index, &input);
        if (err_0 == nullptr) {
          const char* input_name;
          const int64_t* shape;
          TRITONSERVER_Error* err_1 = TRITONBACKEND_InputProperties(
                input, &input_name, nullptr, &shape, nullptr, nullptr, nullptr);
          std::string input_name_str = std::string(input_name);
          if (err_1 == nullptr) {
            if (input_name_str != "START" && input_name_str != "END"
                && input_name_str != "READY") {
              total_batch_size += shape[real_batch_dim];
              break;
            }
            index++;
          }
          else {
            RequestsRespondWithError(requests, request_count, err_1);
            return;
          }
        }
        else {
          RequestsRespondWithError(requests, request_count, err_0);
          return;
        }
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN, (std::string("get total batch_size = ") +
                              std::to_string(total_batch_size))
                                 .c_str());

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)
    && ! sequence_batching_enabled) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  //
  // When operating in the decoupled mode, responses should be created
  // from factories. Here, we instantiate a factory for each request and
  // generate the first response. At each new result from the model the
  // generated response is filled, sent, and another response is created
  // from the factory. The last response is send just like in the
  // non-decoupled mode.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  std::vector<TRITONBACKEND_ResponseFactory*> factories;

  for (size_t i = 0; i < request_count; i++) {
    if (is_decoupled_) {
      TRITONBACKEND_ResponseFactory* factory;
      auto err = TRITONBACKEND_ResponseFactoryNew(&factory, requests[i]);
      if (err == nullptr) {
        factories.emplace_back(factory);
      } else {
        factories.emplace_back(nullptr);
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response factory");
        TRITONSERVER_ErrorDelete(err);
      }
    } else {
      TRITONBACKEND_Response* response;
      auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
      if (err == nullptr) {
        responses.emplace_back(response);
      } else {
        responses.emplace_back(nullptr);
        LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
        TRITONSERVER_ErrorDelete(err);
      }
    }
  }

  std::vector<const char*> input_names;
  std::shared_ptr<std::unordered_map<std::string, Tensor>> input_tensors =
      std::make_shared<std::unordered_map<std::string, Tensor>>();
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
  if (is_decoupled_) {
    generate_response_placeholders(&responses, &factories);
  }
  BackendInputCollector collector(
      requests, request_count, &responses, model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), CudaStream());
  SetInputTensors(
      total_batch_size, requests, request_count, &responses, &collector,
      &input_names, &input_tensors, &input_memories, &cuda_copy);

  // Wait for any in-flight input tensor copies to complete.
#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(CudaStream());
  }
#endif

  uint64_t compute_start_ns = 0;
  SET_TIMESTAMP(compute_start_ns);

  stream_callback_ctx_t context = {total_batch_size, requests,   request_count,
                                   &responses,       &factories, this};

  auto output_tensors =
      Execute(&responses, &context, request_count, input_tensors);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  ReadOutputTensors(
      total_batch_size, output_tensors, requests, request_count, &responses);

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("get response size = ") + std::to_string(responses.size()))
          .c_str());

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send FasterTransformer backend response");
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, (std::string("response is sent")).c_str());
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, (std::string("response is nullptr")).c_str());
    }
  }

  // Report statistics for each request.
  for (uint32_t r = 0; r < request_count; ++r) {
    auto& request = requests[r];
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request,
            (responses[r] != nullptr) /* success */, exec_start_ns,
            compute_start_ns, compute_end_ns, exec_end_ns),
        "failed reporting request statistics");

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, exec_start_ns,
          compute_start_ns, compute_end_ns, exec_end_ns),
      "failed reporting batch request statistics");
}

void
streaming_callback(
    std::shared_ptr<std::unordered_map<std::string, Tensor>> output_tensors,
    void* ctx)
{
  stream_callback_ctx_t* context =
      reinterpret_cast<stream_callback_ctx_t*>(ctx);
  ModelInstanceState* model =
      reinterpret_cast<ModelInstanceState*>(context->model);

  std::vector<TRITONBACKEND_Response*>* responses = context->responses;

  model->ReadOutputTensors(
      context->total_batch_size, output_tensors, context->requests,
      context->request_count, responses);

  for (auto& response : *responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(response, 0, nullptr),
          "failed to send FasterTransformer backend response");
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("streaming response is sent")).c_str());
    } else {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("streaming response is nullptr")).c_str());
    }
  }
  responses->clear();
  generate_response_placeholders(responses, context->factories);
}

int
ThreadForward(
    std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance,
    std::shared_ptr<std::unordered_map<std::string, Tensor>>* input_tensors,
    std::shared_ptr<std::unordered_map<std::string, Tensor>>* output_tensors,
    const int device_id, const int use_stream_cb,
    stream_callback_ctx_t* context)
{
  ft::check_cuda_error(cudaSetDevice(device_id));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Start to forward")).c_str());
  if (use_stream_cb) {
    (*ft_model_instance)->registerCallback(streaming_callback, (void*)context);
  }
  *output_tensors = (*ft_model_instance)->forward(*input_tensors);
  if (use_stream_cb) {
    (*ft_model_instance)->unRegisterCallback();
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string("Stop to forward")).c_str());

  return 0;
}

void
triton_check_inputs(
    std::shared_ptr<std::unordered_map<std::string, Tensor>> output_tensors,
    const char* filename)
{
  auto& output = output_tensors->at("output_ids");
  auto shape = output.shape;
  assert(shape.size() == 3);
  assert(output.type == TYPE_UINT32);
  auto batch_size = shape[0];
  auto length = shape[2];
  std::string fName = filename;
  auto file = std::ofstream(fName, std::ios::out);
  if (!file.is_open()) {
  } else {
    for (size_t i = 0; i < batch_size; i++) {
      for (size_t j = 0; j < length; j++) {
        file << ((uint32_t*)output.data)[i * length + j] << " ";
      }
      file << std::endl;
    }
  }
}

void
ModelInstanceState::BroadcastInputTensors(
    std::shared_ptr<std::unordered_map<std::string, Tensor>>* input_tensors)
{
  int node_id = ft::mpi::getCommWorldRank();

  uint32_t input_count = node_id ? 0 : (*input_tensors)->size();
  ft::mpi::bcast(&input_count, 1, ft::mpi::MPI_TYPE_UINT32_T, 0, ft::mpi::COMM_WORLD);
  if (input_count > bcast_buffers.size()) {
    bcast_buffers.resize(input_count);
  }

  if (node_id) {
    for (uint input_index = 0; input_index < input_count; input_index++) {
      std::vector<size_t> batchn_shape;
      int64_t shape_size = 0;
      int64_t buffer_size = 1;
      ft::mpi::bcast(&shape_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      for (int s_id = 0; s_id < shape_size; s_id++) {
        int64_t val;
        ft::mpi::bcast(&val, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
        batchn_shape.push_back(val);
        buffer_size *= val;
      }
      int64_t data_type_size = 1;
      ft::mpi::bcast(&data_type_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      buffer_size *= data_type_size;
      bcast_buffers[input_index] = (char*) realloc(bcast_buffers[input_index], buffer_size);
      char* input_buffer = bcast_buffers[input_index];
      ft::mpi::bcast(input_buffer, buffer_size, ft::mpi::MPI_TYPE_BYTE, 0, ft::mpi::COMM_WORLD);

      int64_t name_size = 0;
      ft::mpi::bcast(&name_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      char char_name[1024] = {0};
      ft::mpi::bcast(char_name, name_size, ft::mpi::MPI_TYPE_CHAR, 0, ft::mpi::COMM_WORLD);
      uint32_t data_type_num = 0;
      ft::mpi::bcast(&data_type_num, 1, ft::mpi::MPI_TYPE_UINT32_T, 0, ft::mpi::COMM_WORLD);
      TRITONSERVER_DataType triton_data_type =
          TRITONSERVER_DataType(data_type_num);

      (*input_tensors)
          ->insert({std::string(char_name),
                    Tensor{TRITONSERVER_MEMORY_CPU, triton_data_type,
                           batchn_shape, input_buffer}});
    }
  } else {
    int input_index = 0;
    for (auto it = (*input_tensors)->begin(); it != (*input_tensors)->end();
         ++it) {
      std::vector<size_t> batchn_shape = it->second.shape;
      int64_t shape_size = batchn_shape.size();
      int64_t buffer_size = 1;
      ft::mpi::bcast(&shape_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      for (int s_id = 0; s_id < shape_size; s_id++) {
        int64_t val = batchn_shape[s_id];
        ft::mpi::bcast(&val, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
        buffer_size *= val;
      }

      ft::Tensor tmp{
          ft::MEMORY_CPU,
          ft::TYPE_BYTES,
          {1},
          nullptr};  // TODO change the getDataTypeByteNum function to static
      int64_t data_type_size = tmp.getTypeSize(
          triton::Tensor::convertTritonTypeToFt(it->second.type));
      ft::mpi::bcast(&data_type_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      buffer_size *= data_type_size;

      ft::mpi::bcast(
          const_cast<void*>(it->second.data), buffer_size, ft::mpi::MPI_TYPE_BYTE, 0,
          ft::mpi::COMM_WORLD);

      std::string name = it->first;
      int64_t name_size = name.size();
      ft::mpi::bcast(&name_size, 1, ft::mpi::MPI_TYPE_INT64_T, 0, ft::mpi::COMM_WORLD);
      bcast_buffers[input_index] = (char*) realloc(bcast_buffers[input_index], name_size);
      char* char_name = bcast_buffers[input_index];
      int64_t length = (int64_t)name.copy(char_name, name_size);
      ft::FT_CHECK(length == name_size);
      ft::mpi::bcast(char_name, name_size, ft::mpi::MPI_TYPE_CHAR, 0, ft::mpi::COMM_WORLD);

      uint32_t data_type_num = (uint32_t)(it->second.type);
      ft::mpi::bcast(&data_type_num, 1, ft::mpi::MPI_TYPE_UINT32_T, 0, ft::mpi::COMM_WORLD);
      input_index ++;
    }
  }
}

std::shared_ptr<std::unordered_map<std::string, Tensor>>
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    stream_callback_ctx_t* context, const uint32_t response_count,
    std::shared_ptr<std::unordered_map<std::string, Tensor>> input_tensors)
{
  try {
    int node_id = ft::mpi::getCommWorldRank();

    if (node_id == 0) {
      // Debug: input array
      // triton_check_inputs(input_tensors, "triton_in");
    }
    if (node_id) {
      input_tensors =
          std::make_shared<std::unordered_map<std::string, Tensor>>();
    }

    ft::mpi::barrier();

    BroadcastInputTensors(&input_tensors);
    std::vector<std::thread> threads;
    std::shared_ptr<std::unordered_map<std::string, Tensor>>
        output_tensors_list[model_instance_gpu_size_];
    for (int gid = model_instance_device_id_start_;
      gid < model_instance_device_id_start_ + model_instance_gpu_size_; gid++)
    {
      int instance_local_id = gid - model_instance_device_id_start_;
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("before ThreadForward " + std::to_string(gid)))
              .c_str());
      threads.push_back(std::thread(
          ThreadForward, &ft_model_instance_[instance_local_id], &input_tensors,
          &output_tensors_list[instance_local_id], gid,
          is_decoupled_ && gid == model_instance_device_id_start_, context));
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN,
          (std::string("after ThreadForward " + std::to_string(gid)))
              .c_str());
    }

    for (auto& t : threads) {
      t.join();
    }

    auto output_tensors = output_tensors_list[0];
    return output_tensors;
  }
  catch (std::exception& ex) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("FasterTransformer execute failure: " + std::string(ex.what()))
                .c_str()));
    return std::shared_ptr<std::unordered_map<std::string, Tensor>>(nullptr);
  }
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>*
        input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();
  bool sequence_batching_enabled = model_state_->SequenceBatchingEnabled();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("get input count = ") + std::to_string(input_count))
          .c_str());

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    input_names->emplace_back(input_name);

    std::string input_name_str = std::string(input_name);

    bool start_end_ready_flag = (input_name_str == "START" || input_name_str == "END"
      || input_name_str == "READY");

    int shape_dims_start = (int) (sequence_batching_enabled && !start_end_ready_flag);

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape + shape_dims_start, input_shape + input_dims_count);
    if (max_batch_size != 0 && !start_end_ready_flag) {
      batchn_shape[0] = total_batch_size;
    }

    std::vector<size_t> batchn_shape_2(
        input_shape + shape_dims_start, input_shape + input_dims_count);
    if (max_batch_size != 0 && !start_end_ready_flag) {
      batchn_shape_2[0] = total_batch_size;
    }

    // The input must be in contiguous CPU/GPU memory.
    const int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    bool device_is_cpu = true;

    std::vector<BackendMemory::AllocationType> alloc_preference;
    if (device_is_cpu) {
      alloc_preference = {BackendMemory::AllocationType::CPU};
    } else {
      alloc_preference = {BackendMemory::AllocationType::GPU_POOL,
                          BackendMemory::AllocationType::GPU};
    }

    BackendMemory* input_memory;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        BackendMemory::Create(
            model_state_->TritonMemoryManager(), alloc_preference,
            device_is_cpu ? 0 : DeviceId(), batchn_byte_size, &input_memory));
    input_memories->push_back(input_memory);

    TRITONSERVER_MemoryType memory_type = input_memory->MemoryType();
    int64_t memory_type_id = input_memory->MemoryTypeId();
    char* input_buffer = input_memory->MemoryPtr();

    collector->ProcessTensor(
        input_name, input_buffer, batchn_byte_size, memory_type,
        memory_type_id);

    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("collect name: ") + input_name +
         " size: " + std::to_string(batchn_byte_size) + " bytes")
            .c_str());
    (*input_tensors)
        ->insert({std::string(input_name),
                  triton::Tensor{TRITONSERVER_MEMORY_CPU, input_datatype,
                                 batchn_shape_2, input_buffer}});
  }

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("the data is in ") +
       (*cuda_copy ? std::string("GPU") : std::string("CPU")))
          .c_str());
  // Finalize...
  *cuda_copy |= collector->Finalize();
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("the data is in ") +
       (*cuda_copy ? std::string("GPU") : std::string("CPU")))
          .c_str());
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size,
    std::shared_ptr<std::unordered_map<std::string, Tensor>> output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      output_stream_);

  bool cuda_copy = false;
  bool sequence_batching_enabled = model_state_->SequenceBatchingEnabled();
  std::vector<std::vector<char>> string_buffers;

  int idx = 0;
  for (auto it = output_tensors->begin(); it != output_tensors->end(); ++it) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string("Get output_tensors ") + std::to_string(idx) +
         std::string(": ") + std::string(it->first))
            .c_str());
    idx++;
    auto& output = it->second;

    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype = output.type;
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN, (std::string("    output_type: ") +
                                TRITONSERVER_DataTypeString(output_dtype))
                                   .c_str());

    const char* output_buffer = static_cast<const char*>(output.data);

    //  Set output shape
    std::vector<int64_t> batchn_shape = sequence_batching_enabled ? std::vector<int64_t>{1} :
      std::vector<int64_t>{};
    std::string batch_shape_str = sequence_batching_enabled ? "    output shape: [1, " :
      "    output shape: [";
    for (uint i = 0; i < output.shape.size(); i++) {
      batchn_shape.push_back(output.shape[i]);
      batch_shape_str = batch_shape_str + std::to_string(output.shape[i]);
      if (i != output.shape.size() - 1) {
        batch_shape_str = batch_shape_str + ", ";
      } else {
        batch_shape_str = batch_shape_str + "]";
      }
    }

    LOG_MESSAGE(TRITONSERVER_LOG_WARN, batch_shape_str.c_str());
    responder.ProcessTensor(
        it->first, output_dtype, batchn_shape, output_buffer,
        TRITONSERVER_MEMORY_GPU, model_instance_device_id_start_);
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(output_stream_);
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("PERFORMED GPU copy: ") +
       (cuda_copy ? std::string("YES") : std::string("NO")))
          .c_str());
}

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  int provided;
  ft::mpi::initThread(nullptr, nullptr, ft::mpi::THREAD_MULTIPLE, &provided);
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        (std::string("Triton TRITONBACKEND API version: ") +
         std::to_string(api_version_major) + "." +
         std::to_string(api_version_minor) + " does not support '" + name +
         "' TRITONBACKEND API version: " +
         std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
         std::to_string(TRITONBACKEND_API_VERSION_MINOR))
            .c_str());
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Create a ModelState object and associate it with the
  // TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: MPI Finalize");

  ft::mpi::finalize();

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  int node_id = ft::mpi::getCommWorldRank();

  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  int model_instance_id = instance_state->GetModelInstanceId();
  int model_instance_count = instance_state->GetModelInstanceCount();

  LOG_MESSAGE(
    TRITONSERVER_LOG_INFO,
    (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
      " (count " + std::to_string(model_instance_count)
      + ")" + " (instance_id " + std::to_string(model_instance_id) + ")").c_str());

  if (node_id) {
    while (true) {
      instance_state->Execute(
          nullptr, nullptr, 0,
          std::shared_ptr<std::unordered_map<std::string, Tensor>>(nullptr));
    }
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::fastertransformer_backend
