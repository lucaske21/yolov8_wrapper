#include "yolov8/onnx_backend.h"

#include "yolov8/postprocess.h"
#include "yolov8/preprocess.h"

#ifdef YOLOV8_HAS_CUDA_PROVIDER
#include <cuda_provider_factory.h>
#endif

#include <filesystem>
#include <stdexcept>

namespace yolov8 {

OnnxYoloBackend::OnnxYoloBackend(const YoloV8Config& config)
    : config_(config),
      env_(ORT_LOGGING_LEVEL_WARNING, "yolov8_wrapper"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}

void OnnxYoloBackend::initialize() {
  if (config_.model_path.empty()) {
    throw std::runtime_error("Model path is empty.");
  }
  if (!std::filesystem::exists(config_.model_path)) {
    throw std::runtime_error("Model file does not exist: " + config_.model_path);
  }
  if (config_.input_width <= 0 || config_.input_height <= 0) {
    throw std::runtime_error("Input width/height must be positive.");
  }

  Ort::SessionOptions session_options;
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  if (config_.device == Device::CUDA) {
#ifdef YOLOV8_HAS_CUDA_PROVIDER
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    throw std::runtime_error(
        "CUDA device requested, but CUDA Execution Provider is unavailable in this build. "
        "Build with GPU ONNX Runtime and CUDA provider headers.");
#endif
  }

  try {
    session_ = std::make_unique<Ort::Session>(env_, config_.model_path.c_str(), session_options);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("Failed to create ONNX Runtime session: ") + e.what());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  const size_t input_count = session_->GetInputCount();
  const size_t output_count = session_->GetOutputCount();
  if (input_count == 0 || output_count == 0) {
    throw std::runtime_error("Model has no input or output nodes.");
  }

  {
    Ort::AllocatedStringPtr input_name_alloc = session_->GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_alloc = session_->GetOutputNameAllocated(0, allocator);
    if (!input_name_alloc || !output_name_alloc) {
      throw std::runtime_error("Failed to fetch model input/output node names.");
    }
    input_name_ = input_name_alloc.get();
    output_name_ = output_name_alloc.get();
  }

  Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
  auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
  input_shape_ = input_tensor_info.GetShape();
  if (input_shape_.size() != 4) {
    throw std::runtime_error("Unsupported model input rank. Expected [N,C,H,W].");
  }
  input_shape_[0] = 1;
  input_shape_[1] = 3;
  input_shape_[2] = config_.input_height;
  input_shape_[3] = config_.input_width;
}

YoloV8Result OnnxYoloBackend::infer(const cv::Mat& image) const {
  if (!session_) {
    throw std::runtime_error("ONNX backend is not initialized.");
  }
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }

  PreprocessContext context;
  std::vector<float> input_tensor_data = preprocess(image, config_, &context);
  if (input_tensor_data.empty()) {
    throw std::runtime_error("Preprocess produced empty tensor.");
  }

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info_, input_tensor_data.data(), input_tensor_data.size(), input_shape_.data(),
      input_shape_.size());

  std::vector<const char*> input_names = {input_name_.c_str()};
  std::vector<const char*> output_names = {output_name_.c_str()};

  std::vector<Ort::Value> outputs;
  try {
    outputs = session_->Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                            output_names.data(), 1);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("ONNX Runtime inference failed: ") + e.what());
  }

  if (outputs.empty() || !outputs[0].IsTensor()) {
    throw std::runtime_error("Inference returned empty outputs or non-tensor output.");
  }

  const Ort::Value& output_tensor = outputs[0];
  auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_shape = output_info.GetShape();
  size_t element_count = output_info.GetElementCount();
  if (element_count == 0) {
    throw std::runtime_error("Inference returned empty output tensor.");
  }

  const float* output_ptr = output_tensor.GetTensorData<float>();
  std::vector<float> output_data(output_ptr, output_ptr + element_count);
  return postprocess(output_data, output_shape, config_, context);
}

}  // namespace yolov8
