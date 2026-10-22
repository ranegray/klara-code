"""The robot as the agent's code sees it: move the gripper, open and close it, ask where it is.

Every call blocks until the robot is done, returns what it measured, and raises a RobotError if it
could not do what was asked. Positions are meters in the base frame: x forward, y left, z up,
from the floor under the cart's center. Angles are degrees.

It talks to servo_bus_driver over ROS, so it runs wherever ROS does (the robot container).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import GripperCommand as GripperTarget
import numpy as np
from rcl_interfaces.msg import ParameterType
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from xlerobot_driver.bus_config import load_bus_config

from klara.kinematics import Arm, Pose

ARM, GRIPPER = 'left_arm', 'left_gripper'  # groups in the bus config
ARM_ACTION = f'{ARM}_controller/follow_joint_trajectory'
GRIPPER_ACTION = f'{GRIPPER}_controller/gripper_cmd'

LINE_SPEED = 0.05  # m/s, the gripper along a straight line
TURN_SPEED = math.radians(30)  # rad/s, the gripper's pitch and roll along a line
JOINT_SPEED = 0.6  # rad/s, the fastest joint on any move (the driver refuses over 2.0)
LIMIT_MARGIN = 0.05  # rad to keep inside each joint's calibrated range, off its hard stop

# After a move the servos stop a little short, mostly from the arm's weight. Re-aim each joint
# past its goal by the amount it stopped short, until every joint is within SETTLE_TOLERANCE.
SETTLE_TOLERANCE = 0.005  # rad, about 2 mm at the gripper
SETTLE_ROUNDS = 2
SETTLE_LIMIT = 0.1  # rad, the most a joint is re-aimed past its goal
SETTLE_PAUSE = 0.3  # s for the arm to come to rest before measuring
SETTLE_DURATION = 0.5  # s for each correction

# The cart, as a box in the base frame (meters), and how far from it the arm must stay.
CART = (np.array([-0.194, -0.234, 0.0]), np.array([0.198, 0.234, 0.775]))
CLEARANCE = 0.02  # m
# The arm's parts between consecutive frames, from the shoulder (Pitch joint) to the tool.
LINKS = ('upper arm', 'forearm', 'wrist', 'gripper')

ANSWER_TIMEOUT = 5.0  # s for servo_bus_driver to answer a request
STALE_JOINT_STATES = 0.5  # s without /joint_states before the robot counts as gone


class RobotError(Exception):
    """The robot could not do what was asked. The message says why."""


class Unreachable(RobotError):
    """No arm configuration puts the gripper at that position and angle."""


class Unsafe(RobotError):
    """The move would bring the arm too close to the cart."""


class Blocked(RobotError):
    """The arm or gripper stopped part way, probably against something."""


@dataclass(frozen=True)
class ToolPose:
    """Where the gripper is: the point between its fingertips, and which way it points.

    pitch: -90 points straight down, 0 level, 90 straight up.
    roll: about the pointing direction. 0 closes the jaws across the direction the arm reaches,
        90 or -90 along it.
    """

    position: np.ndarray  # (x, y, z) meters
    pitch: float  # degrees
    roll: float  # degrees


@dataclass(frozen=True)
class MoveResult(ToolPose):
    """Where the gripper ended up after a move."""

    error: float  # meters from the target position
    path: str  # 'line', or 'joints' if no straight line was possible and the path curved


@dataclass(frozen=True)
class GripResult:
    holding: bool  # the jaws stopped on something before closing fully
    opening: float  # degrees the jaws are open; 0 is fully closed


class Robot:
    """The left arm and its gripper. Use it in a with block, or call close() when done."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self._node = Node('klara_robot')
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = threading.Thread(target=self._executor.spin, daemon=True)
        self._spinning.start()
        try:
            self._connect(timeout)
        except BaseException:
            self.close()
            raise

    def _connect(self, timeout: float) -> None:
        node = self._node
        self._lock = threading.Lock()
        self._positions: dict[str, float] = {}
        self._received: float | None = None  # time.monotonic() of the latest joint states
        node.create_subscription(JointState, 'joint_states', self._on_joint_states, 10)
        self._trajectory = ActionClient(node, FollowJointTrajectory, ARM_ACTION)
        self._gripper = ActionClient(node, GripperCommand, GRIPPER_ACTION)
        for client in (self._trajectory, self._gripper):
            if not client.wait_for_server(timeout):
                raise RobotError('servo_bus_driver is not running, or its torque is off')

        # The driver knows the real joint ranges (from the servos) and which bus config it runs.
        parameters = AsyncParameterClient(node, 'servo_bus_driver')
        if not parameters.wait_for_services(timeout):
            raise RobotError('servo_bus_driver is not answering parameter requests')
        config_value, = _wait_for(parameters.get_parameters(['config']), ANSWER_TIMEOUT).values
        config = load_bus_config(Path(config_value.string_value))
        self.poses = config.poses  # named poses, joint name -> radians
        arm_joints = [j.name for j in config.group(ARM)]
        self._jaw, = (j.name for j in config.group(GRIPPER))
        names = [f'limits.{name}' for name in (*arm_joints, self._jaw)]
        values = _wait_for(parameters.get_parameters(names), ANSWER_TIMEOUT).values
        if any(v.type != ParameterType.PARAMETER_DOUBLE_ARRAY for v in values):
            raise RobotError('servo_bus_driver does not publish joint limits; restart it')
        limits = {name.removeprefix('limits.'): tuple(v.double_array_value)
                  for name, v in zip(names, values)}

        description = Path(get_package_share_directory('xlerobot_description'))
        self._arm = Arm(description / 'urdf' / 'xlerobot.urdf', limits={
            name: (lo + LIMIT_MARGIN, hi - LIMIT_MARGIN) for name, (lo, hi) in limits.items()})
        if sorted(self._arm.joint_names) != sorted(arm_joints):
            raise RobotError(f'the URDF arm {self._arm.joint_names} is not the bus config '
                             f'{ARM} group {arm_joints}')
        # The jaw closes toward the low end of its range.
        self._jaw_closed = limits[self._jaw][0]
        self._jaw_open = limits[self._jaw][1] - LIMIT_MARGIN

        deadline = time.monotonic() + timeout
        while self._received is None:
            if time.monotonic() > deadline:
                raise RobotError('no joint states from servo_bus_driver')
            time.sleep(0.05)

    def close(self) -> None:
        self._executor.shutdown()
        self._spinning.join()  # it may still be handling a message for the node
        self._node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()

    def __enter__(self) -> Robot:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # What the agent's code calls.

    def get_pose(self) -> ToolPose:
        """Where the gripper is now."""
        pose = self._arm.pose(self._arm_joints())
        return ToolPose(pose.position, math.degrees(pose.pitch), math.degrees(pose.roll))

    def move_to(self, position, pitch: float = -90.0, roll: float = 0.0) -> MoveResult:
        """Move the gripper to a position and angle, in a straight line where possible.

        position: (x, y, z) in meters, for the point between the fingertips.
        pitch: degrees. -90 points the gripper straight down, 0 level, 90 straight up.
        roll: degrees about the pointing direction. 0 closes the jaws across the direction the arm
            reaches, 90 or -90 along it.

        The gripper travels in a straight line at 5 cm/s. If there is no straight line from where
        it is, all joints move at once instead and the path curves; the result says which.
        Raises Unreachable if the arm can't put the gripper there, Unsafe if the move would bring
        the arm near the cart, and Blocked if the arm stopped part way.
        """
        target = Pose(np.asarray(position, dtype=float), math.radians(pitch), math.radians(roll))
        q = self._arm_joints()

        plan = self._arm.line(q, target)
        if plan.ok and self._too_close(plan.joints) is None:
            waypoints, path = plan.joints, 'line'
            times = self._line_times(q, waypoints)
        else:
            solution = self._arm.ik(target, seed=q)
            if not solution.ok:
                raise Unreachable(
                    f'the gripper cannot be at {_format(target.position)} with pitch '
                    f'{pitch:.0f} and roll {roll:.0f}. The closest the arm gets is '
                    f'{solution.position_error * 1000:.0f} mm away, pitch off by '
                    f'{math.degrees(solution.pitch_error):.0f} and roll off by '
                    f'{math.degrees(solution.roll_error):.0f} degrees.')
            if problem := self._too_close(_interpolate(q, solution.joints)):
                raise Unsafe(problem)
            waypoints, path = [solution.joints], 'joints'
            times = [_joint_move_time(q, solution.joints)]

        self._follow(waypoints, times)
        reached = self._arm.pose(self._settle(waypoints[-1]))
        return MoveResult(position=reached.position, pitch=math.degrees(reached.pitch),
                          roll=math.degrees(reached.roll), path=path,
                          error=float(np.linalg.norm(reached.position - target.position)))

    def open_gripper(self) -> GripResult:
        """Open the jaws fully. Raises Blocked if something stops them."""
        result = self._grip(self._jaw_open)
        if not result.reached_goal:
            raise Blocked(f'the jaws stopped {self._opening(result.position):.0f} degrees open, '
                          'short of fully open')
        return GripResult(holding=False, opening=self._opening(result.position))

    def close_gripper(self) -> GripResult:
        """Close the jaws until they shut or stop on something between them.

        holding says whether they stopped on something. Closing on nothing is not an error.
        """
        result = self._grip(self._jaw_closed)
        return GripResult(holding=result.stalled, opening=self._opening(result.position))

    # For the harness and tests, not the agent.

    def move_joints(self, positions: dict[str, float]) -> ToolPose:
        """Move the arm's joints to positions (radians, every arm joint), all at once."""
        q = self._arm_joints()
        goal = np.array([positions[name] for name in self._arm.joint_names])
        if problem := self._too_close(_interpolate(q, goal)):
            raise Unsafe(problem)
        self._follow([goal], [_joint_move_time(q, goal)])
        self._settle(goal)
        return self.get_pose()

    # Talking to the driver.

    def _on_joint_states(self, msg: JointState) -> None:
        with self._lock:
            self._positions.update(zip(msg.name, msg.position))
            self._received = time.monotonic()

    def _arm_joints(self) -> np.ndarray:
        with self._lock:
            if time.monotonic() - self._received > STALE_JOINT_STATES:
                raise RobotError('joint states stopped arriving; is servo_bus_driver running?')
            return np.array([self._positions[name] for name in self._arm.joint_names])

    def _follow(self, waypoints: list[np.ndarray], times: list[float]) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self._arm.joint_names
        goal.trajectory.points = [
            JointTrajectoryPoint(positions=[float(v) for v in q], time_from_start=_duration(t))
            for q, t in zip(waypoints, times)]
        result = self._run(self._trajectory, goal, times[-1])
        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            return
        # The driver reports a stop for its own reasons (a fault, shutting down) with the same
        # code as the arm falling behind its path, so tell them apart by the message.
        if result.error_string.startswith(('hardware fault', 'driver is shutting down')):
            raise RobotError(result.error_string)
        if result.error_code in (FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED,
                                 FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED):
            raise Blocked(f'the arm stopped part way: {result.error_string}')
        raise RobotError(f'servo_bus_driver refused the move: {result.error_string}')

    def _grip(self, target: float) -> GripperCommand.Result:
        goal = GripperCommand.Goal(command=GripperTarget(position=target))
        with self._lock:
            duration = abs(target - self._positions[self._jaw]) / JOINT_SPEED
        result = self._run(self._gripper, goal, duration)
        if not (result.reached_goal or result.stalled):
            raise RobotError('the gripper did not finish moving')
        return result

    def _run(self, client: ActionClient, goal, duration: float):
        """Send a goal and wait for its result, allowing for the driver's extra time to settle."""
        handle = _wait_for(client.send_goal_async(goal), ANSWER_TIMEOUT)
        if not handle.accepted:
            raise RobotError('servo_bus_driver turned the goal down')
        try:
            return _wait_for(handle.get_result_async(), duration + ANSWER_TIMEOUT).result
        except BaseException:
            # Interrupted or timed out: stop the motion rather than leave it running.
            handle.cancel_goal_async()
            raise

    # Planning.

    def _settle(self, goal: np.ndarray) -> np.ndarray:
        """Correct for joints stopping short of goal. Returns where they ended up."""
        time.sleep(SETTLE_PAUSE)
        q, aim = self._arm_joints(), goal
        for _ in range(SETTLE_ROUNDS):
            if np.abs(goal - q).max() <= SETTLE_TOLERANCE:
                break
            aim = np.clip(aim + (goal - q), goal - SETTLE_LIMIT, goal + SETTLE_LIMIT)
            self._follow([np.clip(aim, self._arm.lower, self._arm.upper)], [SETTLE_DURATION])
            time.sleep(SETTLE_PAUSE)
            q = self._arm_joints()
        return q

    def _line_times(self, q: np.ndarray, waypoints: list[np.ndarray]) -> list[float]:
        """When to reach each waypoint, so no speed limit is exceeded along the way."""
        times, t, before = [], 0.0, (q, self._arm.pose(q))
        for w in waypoints:
            pose = self._arm.pose(w)
            t += max(np.linalg.norm(pose.position - before[1].position) / LINE_SPEED,
                     abs(pose.pitch - before[1].pitch) / TURN_SPEED,
                     abs(math.remainder(pose.roll - before[1].roll, math.tau)) / TURN_SPEED,
                     np.abs(w - before[0]).max() / JOINT_SPEED,
                     0.05)
            times.append(t)
            before = (w, pose)
        return times

    def _too_close(self, configurations: list[np.ndarray]) -> str | None:
        """What comes within CLEARANCE of the cart along these joint configurations, if anything."""
        low, high = CART[0] - CLEARANCE, CART[1] + CLEARANCE
        for q in configurations:
            frames = self._arm.frames(q)
            points = [frames[name][:3, 3] for name in (*self._arm.joint_names[1:], 'tool')]
            for link, a, b in zip(LINKS, points, points[1:]):
                for p in np.linspace(a, b, 6):
                    if np.all(p > low) and np.all(p < high):
                        return (f'the move would bring the {link} within '
                                f'{CLEARANCE * 100:.0f} cm of the cart, at {_format(p)}')
        return None

    def _opening(self, jaw: float) -> float:
        return math.degrees(jaw - self._jaw_closed)


def _interpolate(start: np.ndarray, end: np.ndarray, step: float = 0.05) -> list[np.ndarray]:
    """Configurations along a joint move, no joint changing more than step between them."""
    n = max(1, math.ceil(np.abs(end - start).max() / step))
    return [start + (end - start) * i / n for i in range(n + 1)]


def _joint_move_time(start: np.ndarray, end: np.ndarray) -> float:
    return max(np.abs(end - start).max() / JOINT_SPEED, 0.5)


def _wait_for(future, timeout: float):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        raise RobotError('servo_bus_driver did not answer in time')
    return future.result()


def _duration(seconds: float) -> Duration:
    return Duration(sec=int(seconds), nanosec=int(seconds % 1 * 1e9))


def _format(position: np.ndarray) -> str:
    return '(' + ', '.join(f'{v:.3f}' for v in position) + ')'
