#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace yolov8 {

enum class Device { CPU, CUDA };

struct Detection {
  int class_id = -1;
  std::string class_name;
  float score = 0.0f;
  cv::Rect box;
};

struct YoloV8Result {
  std::vector<Detection> detections;
};

}  // namespace yolov8
