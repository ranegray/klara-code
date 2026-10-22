"""Arm kinematics from the URDF: where the tool is for given joint angles, and the reverse.

A pose here is what a 5-joint arm (base yaw, three bending joints, wrist roll) can control:
- position: the tool point, in meters, in the base frame
- pitch: how steeply the gripper points, in radians, from -pi/2 (straight down) through 0 (level)
  to pi/2 (straight up)
- roll: the gripper's rotation about its pointing direction, in radians, measured from the arm's
  bending axis (the axis its pitch, elbow and wrist joints turn about). At 0 the jaws close along
  that axis.

Both angles come from the arm itself rather than from the position, because the arm's bending
plane sits a few centimeters to the side of the base's rotation axis.

Everything here is radians and meters. The agent-facing API converts from degrees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

UP = np.array([0.0, 0.0, 1.0])

# 1 radian of pitch or roll error counts like 5 cm of position error.
_ANGLE_WEIGHT = 0.05
# Keeps steps small near configurations where a joint has little effect (singularities).
_DAMPING = 0.01


@dataclass(frozen=True)
class Pose:
    position: np.ndarray  # (3,) meters
    pitch: float  # radians
    roll: float  # radians


@dataclass(frozen=True)
class IkResult:
    joints: np.ndarray  # radians, in Arm.joint_names order
    ok: bool  # within tolerance of the target
    position_error: float  # meters
    pitch_error: float  # radians
    roll_error: float  # radians


@dataclass(frozen=True)
class LinePlan:
    joints: list[np.ndarray]  # one configuration per waypoint, excluding the start
    ok: bool
    reached: float  # fraction of the line that could be followed, 0..1
    message: str


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    type: str
    origin: np.ndarray  # (4, 4) parent link to joint frame
    axis: np.ndarray  # (3,) in the joint frame
    lower: float
    upper: float


class Arm:
    """One arm of a URDF, from a base frame to a tool frame."""

    def __init__(self, urdf: Path, base: str = 'base_link', tool: str = 'tcp_L',
                 limits: dict[str, tuple[float, float]] | None = None) -> None:
        joints = {j.child: j for j in _parse_joints(urdf)}
        base_path = _path_from_root(joints, base)
        if any(j.type != 'fixed' for j in base_path):
            raise ValueError(f'{base} must be fixed relative to the URDF root')
        self._to_base = np.linalg.inv(_chain(base_path))
        self._tool_path = _path_from_root(joints, tool)
        moving = [j for j in self._tool_path if j.type != 'fixed']
        self.joint_names = [j.name for j in moving]
        self._axes = {j.name: j.axis for j in moving}
        limits = limits or {}
        self.lower = np.array([limits.get(j.name, (j.lower, j.upper))[0] for j in moving])
        self.upper = np.array([limits.get(j.name, (j.lower, j.upper))[1] for j in moving])
        # The first joint turns the whole arm; the second sets the axis it bends about.
        self._yaw_joint, self._bend_joint = moving[0].name, moving[1].name

    # Forward kinematics: joint angles to poses.

    def frames(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Each joint's frame and the tool's (4x4, in the base frame) for joint angles q."""
        angles = dict(zip(self.joint_names, q))
        frames, t = {}, self._to_base
        for j in self._tool_path:
            t = t @ j.origin
            frames[j.name] = t
            if j.type != 'fixed':
                t = t @ _rotation(j.axis, angles[j.name])
        frames['tool'] = t
        return frames

    def pose(self, q: np.ndarray) -> Pose:
        frames = self.frames(q)
        tool = frames['tool']
        pitch = math.asin(np.clip(tool[:3, 2] @ UP, -1.0, 1.0))
        return Pose(tool[:3, 3].copy(), pitch, self._roll(frames))

    def _roll(self, frames: dict[str, np.ndarray]) -> float:
        bend = frames[self._bend_joint][:3, :3] @ self._axes[self._bend_joint]
        approach, jaw = frames['tool'][:3, 2], frames['tool'][:3, 0]
        return math.atan2(np.cross(bend, jaw) @ approach, jaw @ bend)

    # Inverse kinematics: poses to joint angles.

    def ik(self, target: Pose, seed: np.ndarray, position_tolerance: float = 1e-3,
           angle_tolerance: float = math.radians(1.0), iterations: int = 100) -> IkResult:
        """Joint angles that put the tool at target, starting the search from seed.

        Damped least squares: repeatedly measure how far the tool is from the target, estimate
        how a small move of each joint changes that (the Jacobian), and move every joint by the
        amounts that best close the gap, staying inside the joint limits. The search from the
        seed comes first; if it fails, other starting points get a turn. Of the answers that
        work, the one closest to seed wins.
        """
        seed = np.asarray(seed, dtype=float)
        results = []
        for i, start in enumerate(self._starts(seed, target)):
            q = np.clip(start, self.lower, self.upper)
            for _ in range(iterations):
                error = self._error(q, target)
                if (np.linalg.norm(error[:3]) <= position_tolerance
                        and np.all(np.abs(error[3:]) <= _ANGLE_WEIGHT * angle_tolerance)):
                    break
                q = np.clip(q + self._step(q, target, error), self.lower, self.upper)
            result = self._result(q, target, position_tolerance, angle_tolerance)
            if result.ok and i == 0:
                return result  # reached from the seed itself, so it's the nearest answer
            results.append(result)
        if solved := [r for r in results if r.ok]:
            return min(solved, key=lambda r: np.abs(r.joints - seed).sum())
        return min(results, key=lambda r: r.position_error)

    def line(self, q_start: np.ndarray, target: Pose, step: float = 0.01,
             angle_step: float = math.radians(5.0)) -> LinePlan:
        """Joint waypoints that move the tool in a straight line from where it is to target."""
        start = self.pose(q_start)
        turn = _wrap(target.roll - start.roll)
        plan = self._line(q_start, start, target, turn, step, angle_step)
        if not plan.ok and turn != 0:
            # Rolling the short way round can run the wrist into its limit when the long way
            # wouldn't.
            longer = self._line(q_start, start, target, turn - math.copysign(2 * math.pi, turn),
                                step, angle_step)
            if longer.ok:
                return longer
        return plan

    def _line(self, q_start: np.ndarray, start: Pose, target: Pose, turn: float, step: float,
              angle_step: float) -> LinePlan:
        n = max(1, math.ceil(max(np.linalg.norm(target.position - start.position) / step,
                                 abs(target.pitch - start.pitch) / angle_step,
                                 abs(turn) / angle_step)))
        joints, q = [], np.asarray(q_start, dtype=float)
        for i in range(1, n + 1):
            a = i / n
            waypoint = Pose(start.position + a * (target.position - start.position),
                            start.pitch + a * (target.pitch - start.pitch),
                            _wrap(start.roll + a * turn))
            result = self.ik(waypoint, q)
            if not result.ok:
                return LinePlan(joints, False, (i - 1) / n,
                                f'no joint solution {100 * a:.0f}% of the way along the line '
                                f'({result.position_error * 1000:.0f} mm short)')
            q = result.joints
            joints.append(q)
        return LinePlan(joints, True, 1.0, '')

    def _error(self, q: np.ndarray, target: Pose) -> np.ndarray:
        """Position error (m) and pitch and roll errors (rad, weighted to compare with meters)."""
        pose = self.pose(q)
        return np.concatenate([
            pose.position - target.position,
            _ANGLE_WEIGHT * np.array([pose.pitch - target.pitch,
                                      _wrap(pose.roll - target.roll)]),
        ])

    def _jacobian(self, q: np.ndarray, target: Pose, eps: float = 1e-6) -> np.ndarray:
        """How the error changes as each joint moves a little, one column per joint."""
        base = self._error(q, target)
        columns = []
        for i in range(len(q)):
            nudged = q.copy()
            nudged[i] += eps
            columns.append((self._error(nudged, target) - base) / eps)
        return np.stack(columns, axis=1)

    def _step(self, q: np.ndarray, target: Pose, error: np.ndarray) -> np.ndarray:
        """Damped least-squares step, skipping joints pinned against a limit they'd push past."""
        jacobian = self._jacobian(q, target)
        free = np.ones(len(q), dtype=bool)
        for _ in range(len(q)):
            j = jacobian * free
            step = j.T @ np.linalg.solve(j @ j.T + _DAMPING**2 * np.eye(len(error)), -error)
            pinned = ((q <= self.lower) & (step < 0)) | ((q >= self.upper) & (step > 0))
            if not (pinned & free).any():
                return step
            free &= ~pinned
        return step

    def _result(self, q: np.ndarray, target: Pose, position_tolerance: float,
                angle_tolerance: float) -> IkResult:
        pose = self.pose(q)
        position_error = float(np.linalg.norm(pose.position - target.position))
        pitch_error = abs(pose.pitch - target.pitch)
        roll_error = abs(_wrap(pose.roll - target.roll))
        ok = (position_error <= position_tolerance and pitch_error <= angle_tolerance
              and roll_error <= angle_tolerance)
        return IkResult(q, ok, position_error, pitch_error, roll_error)

    def _starts(self, seed: np.ndarray, target: Pose) -> list[np.ndarray]:
        """Starting points: the seed, then other configurations, each also turned to the target.

        The base joint alone decides which way the arm faces, so starting with it already
        turned toward the target leaves the search only the bending joints and the roll.
        """
        rng = np.random.default_rng(0)  # the same starting points every time: repeatable answers
        others = [(self.lower + self.upper) / 2,
                  *(rng.uniform(self.lower, self.upper) for _ in range(6))]
        return [seed, *(self._face(q, target.position) for q in [seed, *others]), *others]

    def _face(self, q: np.ndarray, point: np.ndarray) -> np.ndarray:
        """q with the base joint turned so the tool is on the same bearing as point."""
        frames = self.frames(q)
        yaw = frames[self._yaw_joint]
        center, axis = yaw[:3, 3], yaw[:3, :3] @ self._axes[self._yaw_joint]

        def bearing(p: np.ndarray) -> float:  # direction from the axis to p, seen from above
            return math.atan2(p[1] - center[1], p[0] - center[0])

        # A positive joint angle turns counterclockwise from above if the axis points up.
        turn = _wrap(bearing(point) - bearing(frames['tool'][:3, 3])) * float(np.sign(axis @ UP))
        faced = q.copy()
        faced[0] = np.clip(q[0] + turn, self.lower[0], self.upper[0])
        return faced


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """4x4 rotation by angle about a unit axis (Rodrigues' formula)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    t = np.eye(4)
    t[:3, :3] = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
    return t


def _origin(xyz: str, rpy: str) -> np.ndarray:
    """4x4 transform from a URDF origin: rotate by roll, pitch, yaw (fixed axes), translate."""
    roll, pitch, yaw = (float(v) for v in rpy.split())
    t = (_rotation(UP, yaw) @ _rotation(np.array([0.0, 1.0, 0.0]), pitch)
         @ _rotation(np.array([1.0, 0.0, 0.0]), roll))
    t[:3, 3] = [float(v) for v in xyz.split()]
    return t


def _parse_joints(urdf: Path) -> list[_Joint]:
    joints = []
    for j in ET.parse(urdf).getroot().findall('joint'):
        origin, axis, limit = j.find('origin'), j.find('axis'), j.find('limit')
        joints.append(_Joint(
            name=j.get('name'),
            parent=j.find('parent').get('link'),
            child=j.find('child').get('link'),
            type=j.get('type'),
            origin=_origin(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0',
                           origin.get('rpy', '0 0 0') if origin is not None else '0 0 0'),
            axis=np.array([float(v) for v in (axis.get('xyz') if axis is not None
                                              else '1 0 0').split()]),
            lower=float(limit.get('lower', -math.inf)) if limit is not None else -math.inf,
            upper=float(limit.get('upper', math.inf)) if limit is not None else math.inf,
        ))
    return joints


def _path_from_root(joints: dict[str, _Joint], link: str) -> list[_Joint]:
    """The joints from the URDF's root link down to link, in order."""
    path = []
    while link in joints:
        path.append(joints[link])
        link = joints[link].parent
    return path[::-1]


def _chain(path: list[_Joint]) -> np.ndarray:
    """Transform through a path of fixed joints."""
    t = np.eye(4)
    for j in path:
        t = t @ j.origin
    return t
