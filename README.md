# yolov8_onnx_cpp

一个基于 **C++17 + OpenCV + ONNX Runtime** 的最小可用 YOLOv8 ONNX 检测 wrapper 骨架，支持：

- CPU 推理
- NVIDIA GPU 推理（通过 ONNX Runtime CUDA Execution Provider）

> 当前仅支持 **YOLOv8 detection ONNX**，不支持 segmentation / pose / classify。
> 这不是 Ultralytics Python `.pt` 直接推理方案。

## 功能特性

- `cv::Mat` 输入，检测框结果输出
- 预处理：letterbox、BGR->RGB、归一化、HWC->CHW
- 兼容输出形状：
  - `[1, 84, 8400]`
  - `[1, 8400, 84]`
- 后处理：置信度过滤、NMS、坐标映射回原图
- 清晰错误处理（模型路径、输出 shape、CUDA provider 等）

## 项目结构

```text
yolov8_onnx_cpp/
├── CMakeLists.txt
├── README.md
├── include/
│   └── yolov8/
│       ├── config.h
│       ├── types.h
│       ├── detector.h
│       ├── preprocess.h
│       ├── postprocess.h
│       └── onnx_backend.h
├── src/
│   ├── detector.cpp
│   ├── preprocess.cpp
│   ├── postprocess.cpp
│   └── onnx_backend.cpp
└── examples/
    └── main.cpp
```

## 依赖项

- CMake >= 3.16
- C++17 编译器
- OpenCV（core/imgproc/imgcodecs/dnn）
- ONNX Runtime C++ API

### ONNX Runtime GPU 版本说明

若需要 `Device::CUDA`，必须使用**带 CUDA provider** 的 ONNX Runtime 发行版/构建。

例如 Linux 下常见目录结构：

```text
onnxruntime-linux-x64-gpu-x.y.z/
├── include/
└── lib/
```

## 构建

```bash
cmake -S . -B build \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j
```

`ONNXRUNTIME_DIR` 下建议包含：

- `include/`
- `lib/`（部分发行版可能是 `lib64/` 或 Windows 的 `lib/Release`）

如果你只做 CPU 推理，也可保持 `ONNXRUNTIME_DIR` 指向 CPU 版本（但 `Device::CUDA` 将报错）。

## 运行示例

```bash
./build/yolov8_example /path/to/yolov8n.onnx /path/to/image.jpg cuda
```

或：

```bash
./build/yolov8_example /path/to/yolov8n.onnx /path/to/image.jpg cpu
```

执行后会：

- 打印检测结果
- 在当前目录输出 `result.jpg`

## 当前限制

- 仅支持单张图片推理
- 仅支持 YOLOv8 detection ONNX 常见输出布局
- 仅实现最小骨架，未覆盖批处理、模型 warmup、多输出复杂变体等

## 后续扩展建议

- 支持 batch 推理
- 支持更完整的模型输出适配策略
- 加入性能 profiling（预处理/推理/后处理耗时）
- 增加单元测试与集成测试
