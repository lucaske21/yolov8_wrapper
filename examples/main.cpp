#include "yolov8/detector.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

yolov8::Device parse_device(const std::string& device_arg) {
  if (device_arg == "cpu" || device_arg == "CPU") {
    return yolov8::Device::CPU;
  }
  if (device_arg == "cuda" || device_arg == "CUDA") {
    return yolov8::Device::CUDA;
  }
  throw std::runtime_error("Unsupported device argument: " + device_arg +
                           ". Use 'cpu' or 'cuda'.");
}

std::filesystem::path resolve_image_path(const std::string& image_arg,
                                         const std::filesystem::path& exe_path) {
  const std::filesystem::path input_path(image_arg);
  if (input_path.is_absolute()) {
    return input_path;
  }

  if (std::filesystem::exists(input_path)) {
    return input_path;
  }

  const std::filesystem::path exe_dir = exe_path.parent_path();
  if (!exe_dir.empty()) {
    const std::filesystem::path from_exe_dir = exe_dir / input_path;
    if (std::filesystem::exists(from_exe_dir)) {
      return from_exe_dir;
    }
  }

  return input_path;
}

cv::Mat load_image_checked(const std::filesystem::path& image_path,
                           const std::string& original_image_arg) {
  if (!std::filesystem::exists(image_path)) {
    throw std::runtime_error("Image file does not exist: " + image_path.string());
  }
  if (!std::filesystem::is_regular_file(image_path)) {
    throw std::runtime_error("Image path is not a regular file: " + image_path.string());
  }

  std::ifstream image_file(image_path, std::ios::binary);
  if (!image_file) {
    throw std::runtime_error("Cannot open image file for reading: " + image_path.string());
  }

  std::vector<unsigned char> encoded((std::istreambuf_iterator<char>(image_file)),
                                     std::istreambuf_iterator<char>());
  if (encoded.empty()) {
    throw std::runtime_error("Image file is empty or unreadable: " + image_path.string());
  }

  cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);
  if (image.empty()) {
    const bool has_reader = cv::haveImageReader(image_path.string());
    throw std::runtime_error("Failed to decode image: " + original_image_arg +
                             " (resolved as: " + image_path.string() +
                             ", bytes: " + std::to_string(encoded.size()) +
                             ", haveImageReader: " + (has_reader ? "true" : "false") +
                             ", working directory: " + std::filesystem::current_path().string() +
                             ")");
  }

  return image;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <image.jpg> <cpu|cuda>" << std::endl;
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string image_path = argv[2];
  const std::string device_arg = argv[3];
  const std::filesystem::path exe_path = argv[0];

  try {
    const std::filesystem::path resolved_image_path = resolve_image_path(image_path, exe_path);
    cv::Mat image = load_image_checked(resolved_image_path, image_path);

    yolov8::YoloV8Config config;
    config.model_path = model_path;
    config.device = parse_device(device_arg);
    config.conf_threshold = 0.25f;
    config.iou_threshold = 0.45f;

    yolov8::YoloV8Detector detector(config);
    detector.initialize();

    const yolov8::YoloV8Result result = detector.infer(image);
    std::cout << "Detections: " << result.detections.size() << std::endl;
    for (const auto& det : result.detections) {
      std::cout << "class_id=" << det.class_id << ", class_name=" << det.class_name
                << ", score=" << det.score << ", box=[" << det.box.x << "," << det.box.y << ","
                << det.box.width << "," << det.box.height << "]" << std::endl;

      cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
      const std::string label = det.class_name + " " + cv::format("%.2f", det.score);
      cv::putText(image, label, cv::Point(det.box.x, std::max(0, det.box.y - 8)),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    const std::string output_file = "result.jpg";
    cv::imwrite(output_file, image);
    std::cout << "Saved visualization: " << output_file << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 2;
  }

  return 0;
}
