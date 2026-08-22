ARG BASE_IMAGE=crpi-1pzq998p9m7w0auy.cn-hangzhou.personal.cr.aliyuncs.com/challengecup/supermarket_sorting_final:client
FROM ${BASE_IMAGE}

# Submission image layered on the official Client environment. It contains
# source and model under /workspace/baseline for a no-bind-mount test.
ARG MODEL_SHA256=5763801f2875491ab8b00c61fd1ed539de221ae3d2712cf93ad2464529365daf

ENV ROS_DOMAIN_ID=99 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    SUPERMARKET_ORDER=official \
    SUPERMARKET_ALLOW_RUNTIME_LAYOUT=0 \
    SUPERMARKET_TEST_ORACLE=0 \
    SUPERMARKET_INVENTORY_GEOMETRY_FALLBACK=0 \
    SUPERMARKET_YOLO_REQUIRE_OFFICIAL_CLASSES=1 \
    SUPERMARKET_YOLO_WEIGHTS=/workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt \
    PYTHONPATH=/workspace/baseline:/workspace/baseline/examples/supermarket_sorting:/workspace/baseline/examples/ros2:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages

WORKDIR /workspace/baseline
COPY . /workspace/baseline

RUN echo "${MODEL_SHA256}  /workspace/baseline/examples/supermarket_sorting/perception/checkpoints/supermarket_multiclass.pt" \
      | sha256sum -c - \
    && chmod +x /workspace/baseline/scripts/*.sh \
    && python3 -m py_compile \
      /workspace/baseline/examples/supermarket_sorting/decision/models.py \
      /workspace/baseline/examples/supermarket_sorting/decision/task_manager.py \
      /workspace/baseline/examples/supermarket_sorting/perception/kele_detect.py \
      /workspace/baseline/examples/supermarket_sorting/supermarket_sorting_client.py \
      /workspace/baseline/examples/supermarket_sorting/supermarket_sorting_decision_client.py

ENTRYPOINT []
CMD ["bash"]
