#include "yolov8/detector.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

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

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx> <image.jpg> <cpu|cuda>" << std::endl;
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string image_path = argv[2];
  const std::string device_arg = argv[3];

  try {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
      throw std::runtime_error("Failed to read image: " + image_path);
    }

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
