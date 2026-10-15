"""Start the robot: the servo bus driver, the RealSense camera, and the robot model.

Arguments:
  enable_torque:=false  publish joint states only, with the arm limp
  rviz:=true            also open RViz showing the model
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    description = Path(get_package_share_directory('xlerobot_description'))
    urdf = (description / 'urdf' / 'xlerobot.urdf').read_text()
    return LaunchDescription([
        DeclareLaunchArgument('enable_torque', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='false'),
        Node(
            package='xlerobot_driver',
            executable='servo_bus_driver',
            output='screen',
            parameters=[{'enable_torque': ParameterValue(
                LaunchConfiguration('enable_torque'), value_type=bool)}],
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
        # Publishes the transform of every link, from the URDF and /joint_states.
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': urdf}],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', str(description / 'rviz' / 'robot.rviz')],
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
