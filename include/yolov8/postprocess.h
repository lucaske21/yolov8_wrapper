#pragma once

#include "yolov8/config.h"
#include "yolov8/preprocess.h"
#include "yolov8/types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace yolov8 {

YoloV8Result postprocess(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape,
                         const YoloV8Config& config, const PreprocessContext& context);

}  // namespace yolov8
