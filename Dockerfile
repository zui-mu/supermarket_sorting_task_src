ARG BASE_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.7.1+cu118
ARG TORCHVISION_VERSION=0.22.1+cu118

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV ROS_DISTRO=humble
ENV ROS_DOMAIN_ID=99
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ENV MUJOCO_GL=egl
ENV SUPERMARKET_HEADLESS=1
ENV SUPERMARKET_USE_GS=1

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    software-properties-common \
    && add-apt-repository universe \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    python-is-python3 \
    libgl1 \
    libgl1-mesa-dev \
    libglew-dev \
    libegl1 \
    libegl1-mesa-dev \
    libgles2 \
    libgles2-mesa-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libglm-dev \
    libosmesa6-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    mesa-utils \
    x11-apps \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME}) main" \
    > /etc/apt/sources.list.d/ros2.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ros-humble-ros-base \
      ros-humble-rclpy \
      ros-humble-ros2cli \
      ros-humble-ros2run \
      ros-humble-ros2topic \
      ros-humble-cv-bridge \
      ros-humble-tf2-ros \
      ros-humble-geometry-msgs \
      ros-humble-nav-msgs \
      ros-humble-sensor-msgs \
      ros-humble-std-msgs \
      ros-humble-rmw-cyclonedds-cpp \
      ros-humble-demo-nodes-cpp \
      ros-humble-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
      "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
      --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace/supermarket_sorting_task
COPY . /workspace/supermarket_sorting_task

RUN python3 -m pip install --no-cache-dir -e ".[gs]" \
    && python3 -m pip install --no-cache-dir "ultralytics==8.0.196" \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && printf '%s\n' \
      '{' \
      '    "file_format_version" : "1.0.0",' \
      '    "ICD" : {' \
      '        "library_path" : "libEGL_nvidia.so.0"' \
      '    }' \
      '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && chmod +x /workspace/supermarket_sorting_task/docker/entrypoint.sh

ENTRYPOINT []
CMD ["bash"]
