#pragma once

#include "yolov8/config.h"
#include "yolov8/onnx_backend.h"
#include "yolov8/types.h"

#include <memory>

namespace yolov8 {

class YoloV8Detector {
 public:
  explicit YoloV8Detector(YoloV8Config config);

  void initialize();
  YoloV8Result infer(const cv::Mat& image) const;

 private:
  YoloV8Config config_;
  std::unique_ptr<OnnxYoloBackend> backend_;
};

}  // namespace yolov8
