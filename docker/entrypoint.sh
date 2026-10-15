#!/bin/bash
# Build the mounted workspace, then run the command with it sourced.
set -e
source /opt/ros/jazzy/setup.bash
cd /ws
colcon build --symlink-install
source install/setup.bash
exec "$@"
