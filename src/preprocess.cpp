#include "yolov8/preprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolov8 {

std::vector<float> preprocess(const cv::Mat& image, const YoloV8Config& config,
                              PreprocessContext* context) {
  if (image.empty()) {
    throw std::runtime_error("Input image is empty.");
  }
  if (!context) {
    throw std::runtime_error("PreprocessContext pointer is null.");
  }
  if (config.input_width <= 0 || config.input_height <= 0) {
    throw std::runtime_error("Invalid input size in config.");
  }

  context->original_width = image.cols;
  context->original_height = image.rows;

  cv::Mat working = image;
  if (working.channels() == 4) {
    cv::cvtColor(working, working, cv::COLOR_BGRA2BGR);
  } else if (working.channels() == 1) {
    cv::cvtColor(working, working, cv::COLOR_GRAY2BGR);
  } else if (working.channels() != 3) {
    throw std::runtime_error("Unsupported image channel count. Expected 1, 3, or 4 channels.");
  }

  cv::Mat resized;
  if (config.letterbox) {
    const float r = std::min(static_cast<float>(config.input_width) / static_cast<float>(working.cols),
                             static_cast<float>(config.input_height) / static_cast<float>(working.rows));
    const int new_w = static_cast<int>(std::round(working.cols * r));
    const int new_h = static_cast<int>(std::round(working.rows * r));
    context->scale = r;
    context->scale_x = r;
    context->scale_y = r;
    context->resized_width = new_w;
    context->resized_height = new_h;

    cv::resize(working, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    const int pad_w = config.input_width - new_w;
    const int pad_h = config.input_height - new_h;
    const int left = pad_w / 2;
    const int right = pad_w - left;
    const int top = pad_h / 2;
    const int bottom = pad_h - top;
    context->pad_x = static_cast<float>(left);
    context->pad_y = static_cast<float>(top);
    cv::copyMakeBorder(resized, resized, top, bottom, left, right, cv::BORDER_CONSTANT,
                       cv::Scalar(114, 114, 114));
  } else {
    context->scale_x = static_cast<float>(config.input_width) / static_cast<float>(working.cols);
    context->scale_y = static_cast<float>(config.input_height) / static_cast<float>(working.rows);
    context->scale = std::min(context->scale_x, context->scale_y);
    context->resized_width = config.input_width;
    context->resized_height = config.input_height;
    context->pad_x = 0.0f;
    context->pad_y = 0.0f;
    cv::resize(working, resized, cv::Size(config.input_width, config.input_height), 0, 0, cv::INTER_LINEAR);
  }

  if (resized.cols != config.input_width || resized.rows != config.input_height) {
    throw std::runtime_error("Preprocess produced unexpected tensor image size.");
  }

  cv::Mat rgb_or_bgr = resized;
  if (config.swap_rb) {
    cv::cvtColor(resized, rgb_or_bgr, cv::COLOR_BGR2RGB);
  }

  cv::Mat float_img;
  rgb_or_bgr.convertTo(float_img, CV_32F);
  if (config.normalize) {
    float_img *= (1.0f / 255.0f);
  }

  const int channels = 3;
  std::vector<float> tensor(
      static_cast<size_t>(channels * config.input_width * config.input_height));
  for (int c = 0; c < channels; ++c) {
    for (int y = 0; y < config.input_height; ++y) {
      const cv::Vec3f* row_ptr = float_img.ptr<cv::Vec3f>(y);
      for (int x = 0; x < config.input_width; ++x) {
        const size_t index = static_cast<size_t>(c * config.input_height * config.input_width +
                                                 y * config.input_width + x);
        tensor[index] = row_ptr[x][c];
      }
    }
  }

  return tensor;
}

}  // namespace yolov8
