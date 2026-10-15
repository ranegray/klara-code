import math
from pathlib import Path

import numpy as np
import pytest

from klara.kinematics import Arm, Pose

URDF = Path(__file__).parents[1] / 'ros2_ws/src/xlerobot_description/urdf/xlerobot.urdf'
# Home pose from config/bus1.yaml, in Arm.joint_names order.
HOME = np.array([1.5417, 3.2352, 3.0496, 0.9618, 1.6414])


@pytest.fixture(scope='module')
def arm():
    return Arm(URDF)


def tabletop_poses(arm, count, seed=2):
    """Poses the arm can reach in front of the robot, gripper level to pointing down."""
    rng = np.random.default_rng(seed)
    while count:
        pose = arm.pose(rng.uniform(arm.lower, arm.upper))
        if (pose.position[0] > 0.12 and 0.70 < pose.position[2] < 1.0
                and math.radians(-95) < pose.pitch < math.radians(-20)):
            count -= 1
            yield pose


def test_chain_is_the_left_arm(arm):
    assert arm.joint_names == ['Rotation_L', 'Pitch_L', 'Elbow_L', 'Wrist_Pitch_L', 'Wrist_Roll_L']


def test_home_matches_robot_state_publisher(arm):
    # tf2_echo base_link tcp_L with robot_state_publisher fed exactly these joint angles.
    assert arm.pose(HOME).position == pytest.approx([0.245, 0.168, 0.837], abs=1e-3)


def test_pitch_is_straight_down_when_pointing_down(arm):
    target = Pose(np.array([0.30, 0.17, 0.80]), pitch=-math.pi / 2, roll=0.0)
    result = arm.ik(target, seed=HOME)
    assert result.ok, result
    assert arm.frames(result.joints)['tool'][:3, 2] == pytest.approx([0, 0, -1], abs=0.02)


def test_ik_converges_from_a_nearby_seed(arm):
    rng = np.random.default_rng(1)
    for _ in range(50):
        q = rng.uniform(arm.lower, arm.upper)
        result = arm.ik(arm.pose(q), seed=q + rng.normal(0, 0.05, len(q)))
        assert result.ok, result


def test_ik_from_home_finds_most_tabletop_poses(arm):
    # From far away the search can miss answers that need a very different arm configuration.
    # Straight-line moves avoid that by always seeding from the previous waypoint.
    found = sum(arm.ik(pose, seed=HOME).ok for pose in tabletop_poses(arm, 40))
    assert found >= 34


def test_unreachable_target_reports_how_far_off(arm):
    result = arm.ik(Pose(np.array([2.0, 0.0, 0.8]), 0.0, 0.0), seed=HOME)
    assert not result.ok
    assert result.position_error > 1.0


def test_line_stays_straight(arm):
    start = arm.pose(HOME)
    target = Pose(start.position + np.array([0.05, 0.0, -0.05]), start.pitch, start.roll)
    plan = arm.line(HOME, target)
    assert plan.ok, plan.message
    a, b = start.position, target.position
    for q in plan.joints:
        p = arm.pose(q).position
        off_line = np.linalg.norm(np.cross(p - a, b - a)) / np.linalg.norm(b - a)
        assert off_line < 2e-3


def test_line_reports_where_it_got_stuck(arm):
    start = arm.pose(HOME)
    target = Pose(start.position + np.array([1.0, 0.0, 0.0]), start.pitch, start.roll)
    plan = arm.line(HOME, target)
    assert not plan.ok
    assert 0.0 < plan.reached < 0.6
