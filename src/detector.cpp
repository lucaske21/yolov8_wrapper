#include "yolov8/detector.h"

#include <stdexcept>
#include <utility>

namespace yolov8 {

YoloV8Detector::YoloV8Detector(YoloV8Config config) : config_(std::move(config)) {}

void YoloV8Detector::initialize() {
  backend_ = std::make_unique<OnnxYoloBackend>(config_);
  backend_->initialize();
}

YoloV8Result YoloV8Detector::infer(const cv::Mat& image) const {
  if (!backend_) {
    throw std::runtime_error("Detector not initialized. Call initialize() before infer().");
  }
  return backend_->infer(image);
}

}  // namespace yolov8
