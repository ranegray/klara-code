"""Start the robot: the servo bus driver and the RealSense camera."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='xlerobot_driver',
            executable='servo_bus_driver',
            output='screen',
            # Leave time to drive the arm home before escalating to SIGTERM and SIGKILL.
            sigterm_timeout='20',
        ),
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            namespace='camera',
            name='camera',
            output='screen',
            # The IMU needs write access to /sys, which containers don't get. Unused for now.
            parameters=[{'enable_gyro': False, 'enable_accel': False}],
        ),
    ])
