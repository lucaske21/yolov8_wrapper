cmake_minimum_required(VERSION 3.16)
project(yolov8_onnx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOLOV8_ENABLE_CUDA_PROVIDER "Enable CUDA execution provider support if available" ON)

find_package(OpenCV REQUIRED COMPONENTS core dnn)

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime root directory (with include/ and lib/)")

find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS
    ${ONNXRUNTIME_DIR}/include
)

find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS
    ${ONNXRUNTIME_DIR}/lib
    ${ONNXRUNTIME_DIR}/lib64
    ${ONNXRUNTIME_DIR}/lib/Release
    ${ONNXRUNTIME_DIR}/runtimes/win-x64/native
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found. Set -DONNXRUNTIME_DIR=/path/to/onnxruntime")
endif()

if(YOLOV8_ENABLE_CUDA_PROVIDER)
  find_path(ONNXRUNTIME_CUDA_PROVIDER_INCLUDE_DIR
    NAMES cuda_provider_factory.h
    HINTS
      ${ONNXRUNTIME_DIR}/include
      ${ONNXRUNTIME_DIR}/include/onnxruntime/core/providers/cuda
  )
endif()

add_library(yolov8_wrapper
  src/detector.cpp
  src/preprocess.cpp
  src/postprocess.cpp
  src/onnx_backend.cpp
)

target_include_directories(yolov8_wrapper
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
  PRIVATE
    ${ONNXRUNTIME_CUDA_PROVIDER_INCLUDE_DIR}
)

if(YOLOV8_ENABLE_CUDA_PROVIDER AND ONNXRUNTIME_CUDA_PROVIDER_INCLUDE_DIR)
  target_compile_definitions(yolov8_wrapper PRIVATE YOLOV8_HAS_CUDA_PROVIDER=1)
endif()

target_link_libraries(yolov8_wrapper
  PUBLIC
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

add_executable(yolov8_example examples/main.cpp)
target_link_libraries(yolov8_example PRIVATE yolov8_wrapper)
