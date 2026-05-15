#include "yolov8/postprocess.h"

#include <opencv2/dnn/dnn.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace yolov8 {
namespace {

struct Candidate {
  cv::Rect box;
  int class_id = -1;
  float score = 0.0f;
};

bool is_supported_output_shape(const std::vector<int64_t>& shape) {
  return shape.size() == 3 && shape[0] == 1 && ((shape[1] == 84) || (shape[2] == 84));
}

std::string class_name_for(const YoloV8Config& config, int class_id) {
  if (class_id >= 0 && class_id < static_cast<int>(config.class_names.size())) {
    return config.class_names[static_cast<size_t>(class_id)];
  }
  return "class_" + std::to_string(class_id);
}

cv::Rect cxcywh_to_rect(float cx, float cy, float w, float h, const PreprocessContext& ctx) {
  float x1 = cx - w * 0.5f;
  float y1 = cy - h * 0.5f;
  float x2 = cx + w * 0.5f;
  float y2 = cy + h * 0.5f;

  if (ctx.scale_x <= 0.0f || ctx.scale_y <= 0.0f) {
    return cv::Rect();
  }
  x1 = (x1 - ctx.pad_x) / ctx.scale_x;
  y1 = (y1 - ctx.pad_y) / ctx.scale_y;
  x2 = (x2 - ctx.pad_x) / ctx.scale_x;
  y2 = (y2 - ctx.pad_y) / ctx.scale_y;

  x1 = std::max(0.0f, std::min(x1, static_cast<float>(ctx.original_width - 1)));
  y1 = std::max(0.0f, std::min(y1, static_cast<float>(ctx.original_height - 1)));
  x2 = std::max(0.0f, std::min(x2, static_cast<float>(ctx.original_width - 1)));
  y2 = std::max(0.0f, std::min(y2, static_cast<float>(ctx.original_height - 1)));

  const int ix1 = static_cast<int>(std::round(x1));
  const int iy1 = static_cast<int>(std::round(y1));
  const int ix2 = static_cast<int>(std::round(x2));
  const int iy2 = static_cast<int>(std::round(y2));

  const int width = std::max(0, ix2 - ix1);
  const int height = std::max(0, iy2 - iy1);
  return cv::Rect(ix1, iy1, width, height);
}

std::vector<Candidate> decode_candidates(const std::vector<float>& output_data,
                                         const std::vector<int64_t>& output_shape,
                                         const YoloV8Config& config,
                                         const PreprocessContext& context) {
  if (!is_supported_output_shape(output_shape)) {
    throw std::runtime_error("Unsupported output shape. Expected [1,84,8400] or [1,8400,84].");
  }
  if (context.scale <= 0.0f || context.scale_x <= 0.0f || context.scale_y <= 0.0f) {
    throw std::runtime_error("Invalid preprocess context scale.");
  }

  const bool channels_first = (output_shape[1] == 84);
  const int64_t num_predictions = channels_first ? output_shape[2] : output_shape[1];
  const int64_t channels = channels_first ? output_shape[1] : output_shape[2];
  if (output_data.size() != static_cast<size_t>(num_predictions * channels)) {
    throw std::runtime_error("Output data size does not match output shape.");
  }

  if (channels < 5) {
    throw std::runtime_error("Invalid output channel size.");
  }

  std::vector<Candidate> candidates;
  candidates.reserve(static_cast<size_t>(num_predictions));

  auto value_at = [&](int64_t pred_idx, int64_t ch_idx) -> float {
    if (channels_first) {
      const size_t index = static_cast<size_t>(ch_idx * num_predictions + pred_idx);
      return output_data[index];
    }
    const size_t index = static_cast<size_t>(pred_idx * channels + ch_idx);
    return output_data[index];
  };

  for (int64_t i = 0; i < num_predictions; ++i) {
    const float cx = value_at(i, 0);
    const float cy = value_at(i, 1);
    const float w = value_at(i, 2);
    const float h = value_at(i, 3);

    float best_score = 0.0f;
    int best_class_id = -1;
    for (int64_t c = 4; c < channels; ++c) {
      const float class_score = value_at(i, c);
      if (class_score > best_score) {
        best_score = class_score;
        best_class_id = static_cast<int>(c - 4);
      }
    }

    if (best_score < config.conf_threshold || best_class_id < 0) {
      continue;
    }

    const cv::Rect box = cxcywh_to_rect(cx, cy, w, h, context);
    if (box.width <= 0 || box.height <= 0) {
      continue;
    }

    candidates.push_back(Candidate{box, best_class_id, best_score});
  }
  return candidates;
}

}  // namespace

YoloV8Result postprocess(const std::vector<float>& output_data, const std::vector<int64_t>& output_shape,
                         const YoloV8Config& config, const PreprocessContext& context) {
  if (output_data.empty()) {
    throw std::runtime_error("Postprocess received empty output data.");
  }

  std::vector<Candidate> candidates = decode_candidates(output_data, output_shape, config, context);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(candidates.size());
  scores.reserve(candidates.size());
  for (const auto& c : candidates) {
    boxes.push_back(c.box);
    scores.push_back(c.score);
  }

  std::vector<int> keep_indices;
  cv::dnn::NMSBoxes(boxes, scores, config.conf_threshold, config.iou_threshold, keep_indices);

  YoloV8Result result;
  result.detections.reserve(keep_indices.size());
  for (int idx : keep_indices) {
    if (idx < 0 || idx >= static_cast<int>(candidates.size())) {
      continue;
    }
    const Candidate& c = candidates[static_cast<size_t>(idx)];
    result.detections.push_back(
        Detection{c.class_id, class_name_for(config, c.class_id), c.score, c.box});
  }
  return result;
}

}  // namespace yolov8
