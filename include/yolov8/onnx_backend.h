#pragma once

#include "yolov8/config.h"
#include "yolov8/types.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace yolov8 {

class OnnxYoloBackend {
 public:
  explicit OnnxYoloBackend(const YoloV8Config& config);
  void initialize();
  YoloV8Result infer(const cv::Mat& image) const;

 private:
  YoloV8Config config_;
  Ort::Env env_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo memory_info_;
  std::string input_name_;
  std::string output_name_;
  std::vector<int64_t> input_shape_;
};

}  // namespace yolov8
