#pragma once

#include "yolov8/config.h"

#include <opencv2/core.hpp>

#include <vector>

namespace yolov8 {

struct PreprocessContext {
  int original_width = 0;
  int original_height = 0;
  int resized_width = 0;
  int resized_height = 0;
  float scale = 1.0f;
  float scale_x = 1.0f;
  float scale_y = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

std::vector<float> preprocess(const cv::Mat& image, const YoloV8Config& config,
                              PreprocessContext* context);

}  // namespace yolov8
