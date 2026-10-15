"""servo_bus_driver: the ROS 2 node that owns one Feetech servo bus.

A dedicated thread runs the bus at a fixed rate: read every servo, advance each active motion along
its trajectory, write every goal position, publish /joint_states. ROS callbacks never touch the
bus; they hand motions to that thread and wait for the outcome.

Each group in the bus config becomes a controller offering one action:
/<group>_controller/follow_joint_trajectory, or /<group>_controller/gripper_cmd for a gripper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Callable

from ament_index_python.packages import get_package_share_directory
from control_msgs.action import FollowJointTrajectory, GripperCommand
import rclpy
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from xlerobot_driver.bus_config import Joint, load_bus_config, RADIANS_PER_STEP
from xlerobot_driver.servo_bus import (
    ACCELERATION, BusError, ServoBus, ServoState, STEPS_PER_TURN, TORQUE_ENABLE)
from xlerobot_driver.trajectory import Trajectory

Result = FollowJointTrajectory.Result
# control_msgs has no error code for "the driver stopped this motion"; error_string says why.
STOPPED = Result.PATH_TOLERANCE_VIOLATED
# Below this speed a gripper counts as stopped, in rad/s.
STILL_VELOCITY = 0.05


class Fault(Exception):
    """The hardware is in a state the driver won't move it in."""


class InvalidGoal(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Motion:
    """One trajectory on some joints, shared by an action callback and the bus thread.

    The bus thread fills in desired, actual and load every tick and sets `done` when the motion
    finishes. Every field is read and written under the driver's lock.
    """

    joints: tuple[Joint, ...]
    trajectory: Trajectory
    path_tolerance: tuple[float, ...] | None  # rad per joint; None skips the check
    goal_tolerance: tuple[float, ...]  # rad per joint
    goal_time_tolerance: float  # seconds after the trajectory ends to get within goal tolerance
    gripper: bool = False
    start: float | None = None  # bus clock, set when the bus thread picks the motion up
    still_since: float | None = None  # gripper: motion time at which it stopped moving
    stalled: bool = False  # gripper: stopped short of its target
    desired: tuple[float, ...] = ()
    actual: tuple[float, ...] = ()
    load: tuple[float, ...] = ()
    cancel_requested: bool = False
    canceled: bool = False
    error_code: int | None = None  # Result.SUCCESSFUL or an error code, once finished
    message: str = ''
    done: threading.Event = field(default_factory=threading.Event)


class ServoBusDriver(Node):

    def __init__(self) -> None:
        super().__init__('servo_bus_driver')
        share = Path(get_package_share_directory('xlerobot_driver'))
        param = self.declare_parameter
        config_path = param('config', str(share / 'config' / 'bus1.yaml')).value
        self._rate = param('rate', 50.0).value  # Hz
        self._max_velocity = param('max_velocity', 2.0).value  # rad/s, any joint
        self._path_tolerance = param('path_tolerance', 0.25).value  # rad
        self._goal_tolerance = param('goal_tolerance', 0.05).value  # rad
        self._goal_time_tolerance = param('goal_time_tolerance', 1.0).value  # s
        self._gripper_velocity = param('gripper_velocity', 2.0).value  # rad/s
        self._gripper_settle_time = param('gripper_settle_time', 0.3).value  # s still = stalled
        self._gripper_timeout = param('gripper_timeout', 3.0).value  # s after its trajectory ends
        self._gripper_squeeze = param('gripper_squeeze', 0.15).value  # rad past where it stalled
        self._home_velocity = param('home_velocity', 0.8).value  # rad/s
        acceleration = param('acceleration', 30).value  # servo's own cap, 0-254; 0 = none
        # False: only publish joint states, and never turn torque on. For moving joints by hand.
        self._enable_torque = param('enable_torque', True).value

        self._config = load_bus_config(Path(config_path))
        self._joints = self._config.joints
        self._lock = threading.Lock()
        self._motions: dict[str, Motion] = {}  # the active motion of each group, or of 'home'
        self._fault: str | None = None
        self._closing = False

        self._bus = ServoBus(self._config.port, self._config.baudrate)
        self._bus.open()
        try:
            self._start_bus(acceleration)
        except BaseException:
            self._bus.close()
            raise

        self._joint_states = self.create_publisher(JointState, 'joint_states', 10)
        callbacks = ReentrantCallbackGroup()
        for group, action in self._config.groups.items():
            if action == 'gripper_command':
                ActionServer(
                    self, GripperCommand, f'{group}_controller/gripper_cmd',
                    execute_callback=lambda handle, g=group: self._execute_gripper(g, handle),
                    cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=callbacks)
            else:
                ActionServer(
                    self, FollowJointTrajectory, f'{group}_controller/follow_joint_trajectory',
                    execute_callback=lambda handle, g=group: self._execute_trajectory(g, handle),
                    cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=callbacks)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='bus', daemon=True)
        self._thread.start()
        if self._enable_torque:
            self.get_logger().info(f'{self._config.name}: torque on, holding; '
                                   f'controllers {list(self._config.groups)}')
        else:
            self.get_logger().info(f'{self._config.name}: enable_torque is false; '
                                   'publishing joint states only')

    def _start_bus(self, acceleration: int) -> None:
        """Read limits and positions. With torque enabled, check it's safe, then hold still."""
        self._step_limits: dict[str, tuple[int, int]] = {}
        self._limits: dict[str, tuple[float, float]] = {}  # rad, low to high
        for j in self._joints:
            cal = self._bus.read_calibration(j.servo_id)
            self._step_limits[j.name] = (cal.min_position, cal.max_position)
            self._limits[j.name] = tuple(sorted(
                (j.to_radians(cal.min_position), j.to_radians(cal.max_position))))

        states = self._bus.read_state(self._config.servo_ids)
        if silent := [j.name for j in self._joints if states[j.servo_id] is None]:
            raise Fault(f'no reply from {silent}')
        self._steps = {j.name: states[j.servo_id].position for j in self._joints}
        self._positions = {j.name: j.to_radians(self._steps[j.name]) for j in self._joints}
        self._velocities = dict.fromkeys(self._positions, 0.0)
        self._loads = dict.fromkeys(self._positions, 0.0)
        self._setpoints = dict(self._positions)

        home = self._config.poses.get('home', {})
        for j in self._joints:
            lo, hi = self._limits[j.name]
            if not lo <= home.get(j.name, math.nan) <= hi:
                raise ValueError(f'home pose needs {j.name} within {lo:.3f}..{hi:.3f} rad')
        if not self._enable_torque:
            if on := [j.name for j in self._joints if self._bus.read(j.servo_id, TORQUE_ENABLE)]:
                self.get_logger().warn(f'torque is still on for {on}; run `servo_bus torque-off` '
                                       'to move them by hand')
            return

        # A servo caps a goal outside its range at the limit, so turning torque on to hold a
        # joint that is outside would snap it to the limit.
        outside = []
        for name, steps in self._steps.items():
            lo, hi = self._step_limits[name]
            if not lo <= steps <= hi:
                outside.append(f'{name} at {steps} (range {lo}..{hi})')
        if outside:
            raise Fault(f'refusing to turn torque on, joints outside their range: '
                        f'{"; ".join(outside)}. Move them back by hand and restart.')
        for j in self._joints:
            self._bus.write(j.servo_id, ACCELERATION, acceleration)
        self._bus.write_goal_positions({j.servo_id: self._steps[j.name] for j in self._joints})
        self._bus.set_torque(self._config.servo_ids, True)

    # The bus thread.

    def _run(self) -> None:
        period = 1.0 / self._rate
        next_tick = time.monotonic()
        while not self._stop.is_set():
            try:
                self._tick(time.monotonic())
            except (Fault, BusError) as e:
                self._latch_fault(str(e))
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay < 0:
                next_tick = time.monotonic()  # overran; don't try to catch up
            self._stop.wait(max(delay, 0.0))

    def _tick(self, now: float) -> None:
        states = self._bus.read_state(self._config.servo_ids)
        if self._fault is None:  # only this thread sets _fault
            self._check(states)

        with self._lock:
            for j in self._joints:
                state = states[j.servo_id]
                if state is None:
                    continue
                self._steps[j.name] = state.position
                self._positions[j.name] = j.to_radians(state.position)
                self._velocities[j.name] = j.sign * state.velocity * RADIANS_PER_STEP
                self._loads[j.name] = j.sign * state.load / 1000  # fraction of max torque
            goals = None
            if self._fault is None and self._enable_torque:
                for key, motion in list(self._motions.items()):
                    if motion.start is None:
                        motion.start = now
                    if self._advance(motion, now - motion.start):
                        del self._motions[key]
                goals = {j.servo_id: self._goal_steps(j) for j in self._joints}
        if goals is not None:  # after a fault, stop writing: the servos hold their last goal
            self._bus.write_goal_positions(goals)
        self._publish_joint_states()

    def _check(self, states: dict[int, ServoState | None]) -> None:
        for j in self._joints:
            state = states[j.servo_id]
            if state is None:
                raise Fault(f'{j.name} (servo {j.servo_id}) stopped answering')
            if state.faults:
                raise Fault(f'{j.name} (servo {j.servo_id}) reports {", ".join(state.faults)}')
            if abs(state.position - self._steps[j.name]) > STEPS_PER_TURN // 2:
                raise Fault(f'{j.name} jumped from {self._steps[j.name]} to {state.position} '
                            'steps; it was likely turned past the end of its range by hand')

    def _advance(self, motion: Motion, t: float) -> bool:
        """Move a motion along to time t. Returns True once it has finished."""
        desired = motion.trajectory.sample(t)
        actual = tuple(self._positions[j.name] for j in motion.joints)
        motion.desired, motion.actual = desired, actual
        motion.load = tuple(self._loads[j.name] for j in motion.joints)
        errors = [abs(d - a) for d, a in zip(desired, actual)]

        if motion.cancel_requested:  # stop where the trajectory had got to
            motion.canceled = True
            return self._finish(motion, Result.SUCCESSFUL, 'canceled')

        if motion.path_tolerance is not None and t < motion.trajectory.duration:
            for j, error, tolerance in zip(motion.joints, errors, motion.path_tolerance):
                if error > tolerance:
                    # Probably blocked. Hold where the joints are rather than keep pushing.
                    self._hold(motion.joints, actual)
                    return self._finish(motion, Result.PATH_TOLERANCE_VIOLATED,
                                        f'{j.name} fell {error:.3f} rad behind its path '
                                        f'(tolerance {tolerance:.3f})')
        self._hold(motion.joints, desired)
        if t < motion.trajectory.duration:
            return False

        if all(e <= tol for e, tol in zip(errors, motion.goal_tolerance)):
            return self._finish(motion, Result.SUCCESSFUL, '')
        timed_out = t >= motion.trajectory.duration + motion.goal_time_tolerance
        if motion.gripper:
            # Stalled means the jaw really stopped short of its target, not that it ran late.
            (j,), (target,), (position,) = motion.joints, desired, actual
            if abs(self._velocities[j.name]) > STILL_VELOCITY:
                motion.still_since = None
            elif motion.still_since is None:
                motion.still_since = t
            motion.stalled = (motion.still_since is not None
                              and t - motion.still_since >= self._gripper_settle_time)
            if not motion.stalled and not timed_out:
                return False
            # Holding something, most likely. Squeeze only a little past where it stopped, so the
            # servo grips without straining at full torque.
            squeeze = min(self._gripper_squeeze, abs(target - position))
            self._setpoints[j.name] = position + math.copysign(squeeze, target - position)
            return self._finish(motion, Result.GOAL_TOLERANCE_VIOLATED,
                                'stalled short of target' if motion.stalled
                                else f'still moving {motion.goal_time_tolerance:.1f} s after '
                                     'its trajectory ended')
        if not timed_out:
            return False
        self._hold(motion.joints, actual)
        j, error = max(zip(motion.joints, errors), key=lambda pair: pair[1])
        return self._finish(motion, Result.GOAL_TOLERANCE_VIOLATED,
                            f'{j.name} ended {error:.3f} rad from its goal')

    def _hold(self, joints: tuple[Joint, ...], positions: tuple[float, ...]) -> None:
        for j, position in zip(joints, positions):
            self._setpoints[j.name] = position

    def _finish(self, motion: Motion, error_code: int, message: str) -> bool:
        motion.error_code, motion.message = error_code, message
        motion.done.set()
        return True

    def _goal_steps(self, j: Joint) -> int:
        lo, hi = self._step_limits[j.name]
        return min(max(j.to_steps(self._setpoints[j.name]), lo), hi)

    def _latch_fault(self, message: str) -> None:
        with self._lock:
            if self._fault is not None:
                return
            self._fault = message
            for motion in self._motions.values():
                self._finish(motion, STOPPED, f'hardware fault: {message}')
            self._motions.clear()
        self.get_logger().error(f'{message}. Holding; no motion until the driver restarts.')

    def _publish_joint_states(self) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        with self._lock:
            msg.name = [j.name for j in self._joints]
            msg.position = [self._positions[name] for name in msg.name]
            msg.velocity = [self._velocities[name] for name in msg.name]
            msg.effort = [self._loads[name] for name in msg.name]  # not N·m: see _tick
        self._joint_states.publish(msg)

    # Action callbacks. They run on executor threads and only meet the bus thread under the lock.

    def _execute_trajectory(self, group: str, handle) -> FollowJointTrajectory.Result:
        goal = handle.request
        joints = self._config.group(group)
        result = FollowJointTrajectory.Result()
        try:
            names = list(goal.trajectory.joint_names)
            if sorted(names) != sorted(j.name for j in joints):
                raise InvalidGoal(Result.INVALID_JOINTS,
                                  f'{group} takes exactly {[j.name for j in joints]}, got {names}')
            if any(len(p.positions) != len(names) for p in goal.trajectory.points):
                raise InvalidGoal(Result.INVALID_GOAL, f'every point needs {len(names)} positions')
            order = [names.index(j.name) for j in joints]
            times = [_seconds(p.time_from_start) for p in goal.trajectory.points]
            rows = [tuple(p.positions[i] for i in order) for p in goal.trajectory.points]
            path_tolerance = _tolerances(goal.path_tolerance, joints, self._path_tolerance)
            goal_tolerance = _tolerances(goal.goal_tolerance, joints, self._goal_tolerance)
            goal_time = _seconds(goal.goal_time_tolerance) or self._goal_time_tolerance
            motion = self._start(group, joints, lambda start: Motion(
                joints, self._plan(joints, start, times, rows),
                path_tolerance, goal_tolerance, goal_time))
        except InvalidGoal as e:
            self.get_logger().warn(f'{group}: rejected trajectory: {e}')
            handle.abort()
            result.error_code, result.error_string = e.code, str(e)
            return result

        self._wait(handle, motion, feedback=True)
        result.error_code, result.error_string = motion.error_code, motion.message
        if motion.canceled:
            handle.canceled()
        elif motion.error_code == Result.SUCCESSFUL:
            handle.succeed()
        else:
            handle.abort()
        return result

    def _execute_gripper(self, group: str, handle) -> GripperCommand.Result:
        joints = self._config.group(group)
        target = handle.request.command.position  # max_effort is ignored for now
        result = GripperCommand.Result()

        def motion(start: tuple[float, ...]) -> Motion:
            duration = max(abs(target - start[0]) / self._gripper_velocity, 0.2)
            return Motion(joints, self._plan(joints, start, [duration], [(target,)]),
                          None, (self._goal_tolerance,), self._gripper_timeout, gripper=True)

        try:
            started = self._start(group, joints, motion)
        except InvalidGoal as e:
            self.get_logger().warn(f'{group}: rejected command: {e}')
            handle.abort()
            return result

        self._wait(handle, started)
        with self._lock:
            if started.actual:
                result.position, result.effort = started.actual[0], started.load[0]
        result.reached_goal = started.error_code == Result.SUCCESSFUL and not started.canceled
        result.stalled = started.stalled
        if started.canceled:
            handle.canceled()
        elif result.reached_goal or result.stalled:  # holding an object is a successful grasp
            handle.succeed()
        else:
            handle.abort()
        return result

    def _start(self, key: str, joints: tuple[Joint, ...],
               make_motion: Callable[[tuple[float, ...]], Motion]) -> Motion:
        """Hand a new motion, starting from the joints' current setpoints, to the bus thread."""
        with self._lock:
            if self._fault is not None:
                raise InvalidGoal(STOPPED, f'hardware fault: {self._fault}')
            if self._closing:
                raise InvalidGoal(STOPPED, 'driver is shutting down')
            if not self._enable_torque:
                raise InvalidGoal(STOPPED, 'torque is disabled (enable_torque is false)')
            if key in self._motions:
                raise InvalidGoal(Result.INVALID_GOAL,
                                  f'{key} is already moving; wait for that result or cancel it')
            motion = make_motion(tuple(self._setpoints[j.name] for j in joints))
            self._motions[key] = motion
            return motion

    def _plan(self, joints: tuple[Joint, ...], start: tuple[float, ...],
              times: list[float], rows: list[tuple[float, ...]]) -> Trajectory:
        """Build a trajectory from start through the waypoints, refusing anything unsafe."""
        try:
            trajectory = Trajectory.through(start, times, rows)
        except ValueError as e:
            raise InvalidGoal(Result.INVALID_GOAL, str(e)) from e
        for j, column in zip(joints, zip(*rows)):
            lo, hi = self._limits[j.name]
            if outside := [p for p in column if not lo <= p <= hi]:
                raise InvalidGoal(Result.INVALID_GOAL, f'{j.name} goes to {outside[0]:.3f} rad, '
                                                       f'outside its range {lo:.3f}..{hi:.3f}')
        if (speed := trajectory.max_speed()) > self._max_velocity:
            raise InvalidGoal(Result.INVALID_GOAL,
                              f'a joint would move at {speed:.2f} rad/s, over the '
                              f'{self._max_velocity:.2f} rad/s limit; give it more time')
        return trajectory

    def _wait(self, handle, motion: Motion, feedback: bool = False) -> None:
        while not motion.done.wait(0.1):
            with self._lock:
                if handle.is_cancel_requested:
                    motion.cancel_requested = True
                desired, actual = motion.desired, motion.actual
            if feedback and actual:
                msg = FollowJointTrajectory.Feedback()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.joint_names = [j.name for j in motion.joints]
                msg.desired = JointTrajectoryPoint(positions=list(desired))
                msg.actual = JointTrajectoryPoint(positions=list(actual))
                msg.error = JointTrajectoryPoint(
                    positions=[d - a for d, a in zip(desired, actual)])
                handle.publish_feedback(msg)

    # Shutdown.

    def shutdown(self) -> None:
        """Go home, then turn torque off. If home can't be reached, keep holding with torque on."""
        if not self._enable_torque:
            self._stop.set()
            self._thread.join()
            self._bus.close()
            return
        with self._lock:
            self._closing = True
            for motion in self._motions.values():
                self._finish(motion, STOPPED, 'driver is shutting down')
            self._motions.clear()
            reason = self._fault and f'hardware fault: {self._fault}'
            if reason is None:
                home = self._config.poses['home']
                start = tuple(self._setpoints[j.name] for j in self._joints)
                target = tuple(home[j.name] for j in self._joints)
                distance = max(abs(b - a) for a, b in zip(start, target))
                duration = max(distance / self._home_velocity, 2.0)
                # A gripper may be holding something, so let it stall on the way home.
                gripper = [self._config.groups[j.group] == 'gripper_command' for j in self._joints]
                going_home = Motion(
                    self._joints, Trajectory.through(start, [duration], [target]),
                    tuple(math.inf if g else self._path_tolerance for g in gripper),
                    tuple(math.inf if g else self._goal_tolerance for g in gripper),
                    self._goal_time_tolerance)
                self._motions['home'] = going_home

        if reason is None:
            self.get_logger().info(f'going home over {duration:.1f} s')
            going_home.done.wait(duration + self._goal_time_tolerance + 2.0)
            with self._lock:
                if going_home.error_code != Result.SUCCESSFUL:
                    reason = going_home.message or 'timed out'
        self._stop.set()
        self._thread.join()
        try:
            if reason is None:
                self._bus.set_torque(self._config.servo_ids, False)
                self.get_logger().info('home; torque off')
            else:
                self.get_logger().error(
                    f'could not go home ({reason}). Leaving torque on, holding. Support the arm '
                    'and run `servo_bus torque-off` to release it.')
        finally:
            self._bus.close()


def _seconds(duration_msg) -> float:
    return Duration.from_msg(duration_msg).nanoseconds / 1e9


def _tolerances(given, joints: tuple[Joint, ...], default: float) -> tuple[float, ...]:
    """Per-joint position tolerances from a goal, falling back to the default where unset."""
    by_name = {t.name: t.position for t in given if t.position > 0}
    return tuple(by_name.get(j.name, default) for j in joints)


def main() -> None:
    # Handle SIGINT/SIGTERM ourselves, so ROS stays up while the arm goes home.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    try:
        node = ServoBusDriver()
    except (Fault, BusError, ValueError) as e:
        get_logger('servo_bus_driver').fatal(str(e))
        rclpy.shutdown()
        sys.exit(1)

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    while not stop.wait(0.5):
        pass
    node.shutdown()
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
