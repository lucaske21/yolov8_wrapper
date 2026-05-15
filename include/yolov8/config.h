#pragma once

#include "yolov8/types.h"

#include <string>
#include <vector>

namespace yolov8 {

struct YoloV8Config {
  std::string model_path;
  Device device = Device::CPU;
  int input_width = 640;
  int input_height = 640;
  float conf_threshold = 0.25f;
  float iou_threshold = 0.45f;
  bool letterbox = true;
  bool swap_rb = true;
  bool normalize = true;
  std::vector<std::string> class_names;
};

}  // namespace yolov8
