FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ONNXRUNTIME_VERSION=1.18.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN curl -L -o onnxruntime.tgz \
    https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz \
    && tar -xzf onnxruntime.tgz \
    && rm onnxruntime.tgz

ENV ONNXRUNTIME_VERSION=${ONNXRUNTIME_VERSION}
ENV ONNXRUNTIME_DIR=/opt/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}

WORKDIR /workspace
COPY . .

RUN cmake -S . -B build \
    -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
    -DYOLOV8_ENABLE_CUDA_PROVIDER=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j"$(nproc)"

CMD ["/bin/bash"]
