# Use the base image from NVIDIA with CUDA, cuDNN, and Ubuntu 22.04
ARG IMAGE_NAME=nvidia/cuda
FROM ${IMAGE_NAME}:11.8.0-devel-ubuntu22.04 AS base

# Set environment variables for cuDNN
ENV NV_CUDNN_VERSION 8.9.6.50
ENV NV_CUDNN_PACKAGE_NAME "libcudnn8"
ENV NV_CUDNN_PACKAGE "libcudnn8=$NV_CUDNN_VERSION-1+cuda11.8"
ENV NV_CUDNN_PACKAGE_DEV "libcudnn8-dev=$NV_CUDNN_VERSION-1+cuda11.8"

# Install cuDNN, Python 3, pip, and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ${NV_CUDNN_PACKAGE} \
    ${NV_CUDNN_PACKAGE_DEV} \
    python3 python3-pip \
    libqt5gui5 \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install \
    tensorflow==2.13.0 \
    dill \
    PyQt5 \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    unidecode \
    statsmodels \
    cupy-cuda11x

# Copy all files from the current directory to /root/invis
COPY . /root/invis/

# Set the working directory
WORKDIR /root/invis/

# Set the default command to run your application
CMD ["python3", "Main.py"]
