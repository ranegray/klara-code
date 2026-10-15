# ROS 2 Jazzy plus the robot's hardware dependencies. The workspace source is mounted rather than
# copied in (see compose.yaml), so this image only changes when a dependency does.
FROM ros:jazzy-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-pip \
        ros-jazzy-control-msgs \
        ros-jazzy-realsense2-camera \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# The Feetech SDK is only on PyPI. ROS uses the system Python, so install it there.
RUN pip3 install --no-cache-dir --break-system-packages feetech-servo-sdk==1.0.0

COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
