// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma warning(push, 0)
#include "fastertransformer/gpt.h"
#pragma warning(pop)
#pragma GCC diagnostic pop

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

#include "fastertransformer/triton_backend/transformer.hpp"
#include "fastertransformer/triton_backend/gpt_triton_backend.hpp"
#include <thread>

//
// PyTorch C++ (LibTorch) Backend that implements the TRITONBACKEND API.
//

namespace triton { namespace backend { namespace pytorch {

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

  // Load a TorchScript model using 'artifact_name' as the name for the
  // TorchScript file. Return in 'model_path' the full path to the
  // TorchScript file, return in 'torch_model' the Torch Module
  // representing the model.
  TRITONSERVER_Error* LoadModel
  (const std::string& artifact_name,
   const int32_t node_id,
   const int32_t device_id,
   const cudaStream_t stream,
   std::string* model_path,
   std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance);

   int GetGpuSize() {return gpu_size;};

 private:
  ModelState(TRITONBACKEND_Model* triton_model);
  TRITONSERVER_Error* AutoCompleteConfig();
  std::shared_ptr<AbstractTransformerModel> ftModel;
  int node_id, gpu_size, world_size;
  ncclComm_t tensor_nccl_comms[8];
  ncclComm_t layer_nccl_comms[8];
  cudaStream_t streams_[8];
  std::vector<ncclUniqueId> nccl_ids;
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
    : BackendModel(triton_model)
{
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));

  triton::common::TritonJson::WriteBuffer buffer;
  ModelConfig().PrettyWrite(&buffer);
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value param;
  model_config_.MemberAsObject("parameters", &param);
  auto param_get = [&] (const char* field) {
    common::TritonJson::Value key;
    std::string value;
    param.MemberAsObject(field, &key);
    key.MemberAsString("string_value", &value);
    return value;
  };
  auto param_get_int = [&] (const char* field) {
    int ret = 0;
    try {
      ret = std::stoi(param_get(field));
    } catch (std::invalid_argument& ia) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  (std::string("Invalid configuration argument '") + field + "': " + ia.what()).c_str());
    }
    return ret;
  };
  auto param_get_float = [&] (const char* field) {
    float ret = 0.0;
    try {
      ret = std::stof(param_get(field));
    } catch (std::invalid_argument& ia) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  (std::string("Invalid configuration argument '") + field + "': " + ia.what()).c_str());
    }
    return ret;
  };

  auto modelVersionPath = JoinPath({RepositoryPath(), std::to_string(Version()), "/"});

  if (param_get_int("is_half"))
    ftModel.reset(new GptModel<fastertransformer::OperationType::FP16>
                  (param_get_int("batch_size"),
                   param_get_int("candidate_num"),
                   param_get_int("head_num"),
                   param_get_int("size_per_head"),
                   param_get_int("vocab_size"),
                   param_get_int("max_seq_len"),
                   param_get_int("decoder_layers"),
                   param_get_int("tensor_para_size"),
                   param_get_int("layer_para_size"),
                   param_get_int("layer_para_batch_size"),
                   param_get_float("probability_threshold"),
                   param_get_int("is_fuse_QKV"),
                   param_get_int("temperature"),
                   param_get_float("repetition_penalty"),
                   param_get("model_name"),
                   modelVersionPath));
  else
    ftModel.reset(new GptModel<fastertransformer::OperationType::FP32>
                  (param_get_int("batch_size"),
                   param_get_int("candidate_num"),
                   param_get_int("head_num"),
                   param_get_int("size_per_head"),
                   param_get_int("vocab_size"),
                   param_get_int("max_seq_len"),
                   param_get_int("decoder_layers"),
                   param_get_int("tensor_para_size"),
                   param_get_int("layer_para_size"),
                   param_get_int("layer_para_batch_size"),
                   param_get_float("probability_threshold"),
                   param_get_int("is_fuse_QKV"),
                   param_get_int("temperature"),
                   param_get_float("repetition_penalty"),
                   param_get("model_name"),
                   modelVersionPath));


  int tensor_para_size = ftModel->get_tensor_para_size();
  int layer_para_size = ftModel->get_layer_para_size();
  world_size = tensor_para_size * layer_para_size;
  
  
  CUDACHECK(cudaGetDeviceCount(&gpu_size));

  assert(tensor_para_size <= gpu_size);

  if (node_id == 0) nccl_ids = ftModel->create_nccl_ids(world_size);

  int nccl_size = nccl_ids.size();
  MPICHECK(MPI_Bcast(&nccl_size, 1, MPI_INT, 0, MPI_COMM_WORLD));
  if(node_id) nccl_ids.resize(nccl_size);
  for(size_t i = 0; i < nccl_ids.size(); i++)
  {
      MPICHECK( MPI_Bcast(&nccl_ids[i], sizeof(nccl_ids[i]), MPI_BYTE, 0, MPI_COMM_WORLD));
  }

  NCCLCHECK(ncclGroupStart());
  for (int gid = 0; gid < gpu_size; gid ++) {

    LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                (std::string("enter nccl group") + std::to_string(gid)).c_str());
    int rank = node_id * gpu_size + gid;
    size_t tensor_para_rank = rank % tensor_para_size;
    size_t layer_para_rank = rank / tensor_para_size;
    ncclUniqueId tensor_para_nccl_uid = nccl_ids[layer_para_rank];
    ncclUniqueId layer_para_nccl_uid  = nccl_ids[layer_para_size + tensor_para_rank];

    CUDACHECK(cudaSetDevice(gid));
    NCCLCHECK( ncclCommInitRank(&tensor_nccl_comms[gid], tensor_para_size, tensor_para_nccl_uid, tensor_para_rank));
    NCCLCHECK( ncclCommInitRank(&layer_nccl_comms[gid], layer_para_size, layer_para_nccl_uid, layer_para_rank));
  }
  NCCLCHECK(ncclGroupEnd());

  LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
              (std::string("Model is loaded as'") + ftModel->to_string()).c_str());
}

TRITONSERVER_Error*
ModelState::LoadModel(
    const std::string& artifact_name,
    const int32_t node_id,
    const int32_t device_id,
    const cudaStream_t stream,
    std::string* model_path,
    std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance)
{
  // Find the TorchScript file that describes the model. If the model
  // configuration doesn't have an explicit model file specified then
  // use the default name ("model.pt").
  CUDACHECK(cudaSetDevice(device_id));
  std::string cc_model_filename = artifact_name;
  if (cc_model_filename.empty()) {
    cc_model_filename = "gpt3-model";
  }

  {
    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    printf("before allocation, free %.2f GB total %.2f GB\n", free, total);
  }

  auto path = JoinPath({RepositoryPath(), std::to_string(Version()), cc_model_filename});

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("Model path ") + path).c_str());

  cudaStreamCreate(&streams_[device_id]);
  auto modelInstance = ftModel->createModelInstance(node_id, device_id, world_size, streams_[device_id]);
  auto param_instance = ftModel->createParamInstance(node_id, device_id, world_size, streams_[device_id], nccl_ids);
  param_instance->init_nccl_from_comms(tensor_nccl_comms[device_id], layer_nccl_comms[device_id]);
  modelInstance->set_param(param_instance.get());
  ft_model_instance->reset(modelInstance.release());

  *model_path = JoinPath(
      {RepositoryPath(), std::to_string(Version()), cc_model_filename});

  {
    size_t free_bytes, total_bytes;
    check_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes));
    float free = (float)(free_bytes) / 1024.0 / 1024.0 / 1024.0;
    float total = (float)(total_bytes) / 1024.0 / 1024.0 / 1024.0;
    printf("after allocation, free %.2f GB total %.2f GB\n", free, total);
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::AutoCompleteConfig()
{
  // Auto-complete configuration is not supported since PyTorch does not
  // store/capture sufficient model metadata so just log error instead.
  LOG_MESSAGE(
      TRITONSERVER_LOG_WARN,
      (std::string("skipping model configuration auto-complete for '") +
       Name() + "': not supported for pytorch backend")
          .c_str());

  return nullptr;  // success
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
  
  std::shared_ptr<std::vector<Tensor>> Execute(
      std::vector<TRITONBACKEND_Response*>* responses,
      const uint32_t response_count,
      std::shared_ptr<std::vector<Tensor>> input_tensors);

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
      std::shared_ptr<std::vector<Tensor>>* input_tensors,
      std::vector<BackendMemory*>* input_memories, bool* cuda_copy);
  void ReadOutputTensors(
      size_t total_batch_size, const std::vector<const char*>& output_names,
      std::shared_ptr<std::vector<Tensor>> output_tensors,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses);

  ModelState* model_state_;

  // The full path to the TorchScript model file.
  std::string model_path_;


  std::unique_ptr<AbstractTransformerModelInstance> ft_model_instance_[8];

  // Map from configuration name for an input to the index of
  // that input in the model.
  std::unordered_map<std::string, int> input_index_map_;

  // Map from configuration name for an output to the index of
  // that output in the model.
  std::unordered_map<std::string, int> output_index_map_;
  std::unordered_map<std::string, TRITONSERVER_DataType> output_dtype_map_;
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

int ThreadLoadModel(ModelState* model_state,
                    const std::string& artifact_name,
                    const int32_t node_id,
                    const int32_t device_id,
                    const cudaStream_t stream,
                    std::string* model_path,
                    std::unique_ptr<AbstractTransformerModelInstance>* ft_model_instance)
{
  THROW_IF_BACKEND_INSTANCE_ERROR
      (model_state->LoadModel
      (artifact_name, node_id, device_id, stream, model_path, ft_model_instance));
  return 0;
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
  int node_id, num_nodes;
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
  MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &num_nodes));
  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("Faster transformer model instance is created at GPU '") +
                std::to_string(DeviceId()) + "'").c_str());


  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("Model name ") + ArtifactFilename()).c_str());

  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateInputs());
  THROW_IF_BACKEND_INSTANCE_ERROR(ValidateOutputs());


  std::vector<std::thread> threads;
  int gpu_size = model_state->GetGpuSize();

  for(int gid = 0; gid < gpu_size; gid ++) {
    threads.push_back(std::thread(ThreadLoadModel,
                                  model_state,
                                  ArtifactFilename(), node_id, gid, CudaStream(),
                                  &model_path_, &ft_model_instance_[gid]));
  }
  for(auto & t : threads) {
    t.join();
  }

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, DeviceId());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Model instance is created on GPU ") + prop.name).c_str());
   
}

ModelInstanceState::~ModelInstanceState()
{
#ifdef TRITON_ENABLE_GPU
#endif  // TRITON_ENABLE_GPU
}

TRITONSERVER_Error*
ModelInstanceState::ValidateInputs()
{
  triton::common::TritonJson::Value ios;
  std::string name, data_type;
  triton::common::TritonJson::Value jshape;
  model_state_->ModelConfig().MemberAsArray("input", &ios);

  for (size_t size = 0; size < ios.ArraySize(); size++){
    triton::common::TritonJson::Value input;
    ios.IndexAsObject(size, &input);
    input.MemberAsString("name", &name);
    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("get input name: " + name).c_str()));
    input.MemberAsString("data_type", &data_type);
    input.MemberAsArray("dims", &jshape);

    std::vector<size_t> shape;
    for(size_t size = 0; size < jshape.ArraySize(); size++){
      size_t value;
      jshape.IndexAsUInt(size, &value);
      shape.push_back(value);
    }

    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("input: ") + name +
                 ", type: " + data_type +
                 ", shape: [" + std::to_string(shape[0]) +  ", " + std::to_string(shape[1]) + "]").c_str());
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
  for (size_t size = 0; size < ios.ArraySize(); size++){
    triton::common::TritonJson::Value input;
    ios.IndexAsObject(size, &input);
    input.MemberAsString("name", &name);
    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("get input name: " + name).c_str()));
    input.MemberAsString("data_type", &data_type);
    input.MemberAsArray("dims", &jshape);

    std::vector<size_t> shape;
    for(size_t size = 0; size < jshape.ArraySize(); size++){
      size_t value;
      jshape.IndexAsUInt(size, &value);
      shape.push_back(value);
    }

    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("input: ") + name +
                 ", type: " + data_type +
                 ", shape: [" + std::to_string(shape[0]) +  ", " + std::to_string(shape[1]) + "]").c_str());
  }

  return nullptr;  // success
}

void
ModelInstanceState::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  int node_id, num_nodes;
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
  MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &num_nodes));

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
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to PyTorch backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("get total batch_size = ") +
               std::to_string(total_batch_size)).c_str());

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
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
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
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

  std::vector<const char*> input_names;
  std::shared_ptr<std::vector<Tensor>> input_tensors = std::make_shared<std::vector<Tensor>>();
  std::vector<BackendMemory*> input_memories;
  bool cuda_copy = false;
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

  auto output_tensors = Execute(&responses, request_count, input_tensors);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // Free BackendMemory used for inputs
  for (BackendMemory* mem : input_memories) {
    delete mem;
  }
  input_memories.clear();

  // Verify output indices are valid with number of outputs after execution
  std::vector<const char*> output_names;
  output_names.push_back("OUTPUT0");
  bool invalid_index = false;
  int max_index = output_tensors->size() - 1;
  for (const auto& name : output_names) {
    int op_index = output_index_map_[name];
    if ((op_index < 0) || (op_index > max_index)) {
      SendErrorForResponses(
          &responses, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              std::string(
                  "The output " + std::string(name) +
                  " in the model configuration refers to an output index which"
                  " doesn't exist. This model has " +
                  std::to_string(max_index + 1) + " outputs")
                  .c_str()));
      invalid_index = true;
      break;
    }
  }

  if (!invalid_index) {
    ReadOutputTensors(
        total_batch_size, output_names, output_tensors, requests, request_count,
        &responses);
  }

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("get response size = ") + std::to_string(responses.size())).c_str());

  // Send all the responses that haven't already been sent because of
  // an earlier error. Note that the responses are not set to nullptr
  // here as we need that indication below to determine if the request
  // we successful or not.
  for (auto& response : responses) {
    if (response != nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
          "failed to send PyTorch backend response");
      LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                  (std::string("response is sent")).c_str());
    }
    else {
      LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                  (std::string("response is nullptr")).c_str());
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

int ThreadForward(std::unique_ptr<AbstractTransformerModelInstance> *ft_model_instance,
                  std::shared_ptr<std::vector<Tensor>> *input_tensors,
                  std::shared_ptr<std::vector<Tensor>> *output_tensors,
                  const int device_id)
{
  CUDACHECK(cudaSetDevice(device_id));
  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("Start to forward")).c_str());
  // output_tensors = ft_model_instance->forward(input_tensors);
  *output_tensors = (*ft_model_instance)->forward(*input_tensors);
  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("Stop to forward")).c_str());

  return 0;
}

void triton_check_inputs(std::shared_ptr<std::vector<Tensor>> output_tensors, const char* filename)
{
  auto& output = output_tensors->at(0);
  auto shape = output.shape;
  assert(shape.size() == 2);
  assert(output.type == TYPE_UINT32);
  auto batch_size = shape[0];
  auto length = shape[1];
  std::string fName = filename;
  auto file = std::ofstream(fName, std::ios::out);
  if(!file.is_open())  {
  } else {
    for(int i=0; i<batch_size; i++) {
      for(int j=0; j<length; j++) {
        file << ((uint32_t*)output.data)[i*length + j] << " ";
      }
      file << std::endl;
    }
  }
}

void BroadcastInputTensors(std::shared_ptr<std::vector<Tensor>>* input_tensors)
{
  int node_id, num_nodes;
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
  MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &num_nodes));

  uint32_t input_count = node_id ? 0 : (*input_tensors)->size();
  MPI_Bcast(&input_count, 1 , MPI_UINT32_T , 0 , MPI_COMM_WORLD);
  if (node_id) 
  {
    for (uint32_t i = 0; i < input_count; ++i)
    {
      std::vector<int64_t> batchn_shape;
      int64_t batch_size, length;
      MPICHECK(MPI_Bcast(&(batch_size), 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
      MPICHECK(MPI_Bcast(&(length), 1, MPI_INT64_T, 0, MPI_COMM_WORLD));
      batchn_shape.push_back(batch_size);
      batchn_shape.push_back(length);

      uint32_t* input_buffer = new uint32_t[batchn_shape[0] * batchn_shape[1]];

      MPICHECK(MPI_Bcast(input_buffer, batchn_shape[0] * batchn_shape[1], MPI_UINT32_T, 0, MPI_COMM_WORLD));

      (*input_tensors)->push_back(Tensor{TRITONSERVER_MEMORY_CPU, TYPE_UINT32, batchn_shape, input_buffer});
    }
  }
  else
  {
    for (uint32_t i = 0; i < input_count; ++i)
    {
      auto batch_size = (*input_tensors)->at(i).shape[0];
      auto length = (*input_tensors)->at(i).shape[1];

      MPICHECK(MPI_Bcast(&(batch_size), 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

      MPICHECK(MPI_Bcast(&(length), 1, MPI_INT64_T, 0, MPI_COMM_WORLD));

      MPICHECK(MPI_Bcast((*input_tensors)->at(i).data, batch_size * length, MPI_UINT32_T, 0, MPI_COMM_WORLD));

    }
  }
}

std::shared_ptr<std::vector<Tensor>>
ModelInstanceState::Execute(
    std::vector<TRITONBACKEND_Response*>* responses,
    const uint32_t response_count,
    std::shared_ptr<std::vector<Tensor>> input_tensors)
{

  try {
    const int gpu_size = model_state_->GetGpuSize();
    int node_id, num_nodes;
    MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
    MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &num_nodes));

    if (node_id == 0) {check_inputs(input_tensors);triton_check_inputs(input_tensors, "triton_in");}
    if (node_id) input_tensors = std::make_shared<std::vector<Tensor>>();

    MPI_Barrier(MPI_COMM_WORLD);

    BroadcastInputTensors(&input_tensors);
    
    std::vector<std::thread> threads;
    std::shared_ptr<std::vector<Tensor>> output_tensors_list[gpu_size];
    for(int gid = 0; gid < gpu_size; gid ++)
    {
      LOG_MESSAGE(TRITONSERVER_LOG_WARN, (std::string("before ThreadForward " + std::to_string(gid))).c_str());
      threads.push_back(std::thread(ThreadForward, &ft_model_instance_[gid], &input_tensors, &output_tensors_list[gid], gid));
      LOG_MESSAGE(TRITONSERVER_LOG_WARN, (std::string("after ThreadForward " + std::to_string(gid))).c_str());
    }
    for(auto & t : threads)
    {
      t.join();
    }

    auto output_tensors = output_tensors_list[0];
    check_outputs(output_tensors);
    return output_tensors;
  }
  catch (std::exception& ex) {
    SendErrorForResponses(
        responses, response_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("PyTorch execute failure: " + std::string(ex.what())).c_str()));
    return std::shared_ptr<std::vector<Tensor>>(nullptr);
  }
}

void
ModelInstanceState::SetInputTensors(
    size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    BackendInputCollector* collector, std::vector<const char*>* input_names,
    std::shared_ptr<std::vector<Tensor>>* input_tensors,
    std::vector<BackendMemory*>* input_memories, bool* cuda_copy)
{
  const int max_batch_size = model_state_->MaxBatchSize();

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("get input count = ") +
               std::to_string(input_count)).c_str());
  
  char const * input_name_order[3] = {"INPUT_ID", "REQUEST_INPUT_LEN", "REQUEST_OUTPUT_LEN"};

  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        TRITONBACKEND_RequestInput(requests[0], input_name_order[input_idx], &input));

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

    // The shape for the entire input patch, [total_batch_size, ...]
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);
    if (max_batch_size != 0) {
      batchn_shape[0] = total_batch_size;
    }

    // The input must be in contiguous CPU/GPU memory.
    const int64_t batchn_byte_size = GetByteSize(input_datatype, batchn_shape);

    bool device_is_cpu = true;

    std::vector<BackendMemory::AllocationType> alloc_perference;
    if (device_is_cpu) {
      alloc_perference = {BackendMemory::AllocationType::CPU};
    } else {
      alloc_perference = {BackendMemory::AllocationType::GPU_POOL,
                          BackendMemory::AllocationType::GPU};
    }

    BackendMemory* input_memory;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        responses, request_count,
        BackendMemory::Create(
            model_state_->TritonMemoryManager(), alloc_perference,
            device_is_cpu ? 0 : DeviceId(), batchn_byte_size,
            &input_memory));
    input_memories->push_back(input_memory);

    TRITONSERVER_MemoryType memory_type = input_memory->MemoryType();
    int64_t memory_type_id = input_memory->MemoryTypeId();
    char* input_buffer = input_memory->MemoryPtr();

    collector->ProcessTensor(
        input_name, input_buffer, batchn_byte_size, memory_type,
        memory_type_id);

    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("collect name: ") + input_name +
                 " size: " + std::to_string(batchn_byte_size)).c_str());
    (*input_tensors)->push_back(Tensor{TRITONSERVER_MEMORY_CPU, input_datatype, batchn_shape, input_buffer});
  }

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("the data is in ") + (*cuda_copy ? std::string("GPU") : std::string("CPU"))).c_str());
  // Finalize...
  *cuda_copy |= collector->Finalize();
  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("the data is in ") + (*cuda_copy ? std::string("GPU") : std::string("CPU"))).c_str());
}

void
ModelInstanceState::ReadOutputTensors(
    size_t total_batch_size, const std::vector<const char*>& output_names,
    std::shared_ptr<std::vector<Tensor>> output_tensors,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses)
{
  BackendOutputResponder responder(
      requests, request_count, responses, model_state_->MaxBatchSize(),
      model_state_->TritonMemoryManager(), model_state_->EnablePinnedInput(),
      CudaStream());

  bool cuda_copy = false;
  std::vector<std::vector<char>> string_buffers;
  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("output name size") + std::to_string(output_names.size()) + ", name: " + output_names[0]).c_str());

  for (size_t idx = 0; idx < output_names.size(); idx++) {
    std::string name = output_names[idx];
    int op_index = output_index_map_[name];
    name = "OUTPUT0";
    op_index = 0;
    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("get output_tensors 0")).c_str());
    auto& output = output_tensors->at(0);


    // Verify output datatype matches datatype from model config
    TRITONSERVER_DataType output_dtype = output.type;
    TRITONSERVER_DataType config_datatype = TRITONSERVER_TYPE_UINT32;
    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("get output_type: ") + TRITONSERVER_DataTypeString(output.type) + ", " + TRITONSERVER_DataTypeString(config_datatype) ).c_str());

    const char* output_buffer = static_cast<const char*>(output.data);

    //  Set output shape
    std::vector<int64_t> batchn_shape(output.shape);

    LOG_MESSAGE(TRITONSERVER_LOG_WARN,
                (std::string("output shape: [") + std::to_string(batchn_shape[0]) + ", " + std::to_string(batchn_shape[1]) + "]").c_str());

    responder.ProcessTensor(
        name, output_dtype, batchn_shape, output_buffer,
        TRITONSERVER_MEMORY_GPU,
        DeviceId());
  }

  // Finalize and wait for any pending buffer copies.
  cuda_copy |= responder.Finalize();

#ifdef TRITON_ENABLE_GPU
  if (cuda_copy) {
    cudaStreamSynchronize(stream_);
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(TRITONSERVER_LOG_WARN,
              (std::string("PERFORMED GPU copy: ") + (cuda_copy ? std::string("YES") : std::string("NO")) ).c_str());

}

/////////////

extern "C" {

TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  int provided;
  MPI_Init_thread( NULL, NULL, MPI_THREAD_MULTIPLE, &provided); 
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

  MPI_Finalize();

  return nullptr;  // success
}

TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

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

  int node_id, num_nodes;
  MPICHECK( MPI_Comm_rank(MPI_COMM_WORLD, &node_id));
  MPICHECK( MPI_Comm_size(MPI_COMM_WORLD, &num_nodes));

  if (node_id)
  {
    while(true)
    {
      instance_state->Execute(nullptr, 0, std::shared_ptr<std::vector<Tensor>>(nullptr));
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

}}}  // namespace triton::backend::pytorch