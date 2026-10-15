"""Bus configuration: which servo drives which URDF joint, and how servo steps map to radians."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import yaml

from xlerobot_driver.servo_bus import STEPS_PER_TURN

# Calibration stores a homing offset on each servo so that this reading means the URDF zero pose.
ZERO_STEPS = STEPS_PER_TURN // 2
RADIANS_PER_STEP = 2 * math.pi / STEPS_PER_TURN

# The actions a group can offer. A gripper group has exactly one joint.
ACTIONS = ('follow_joint_trajectory', 'gripper_command')


@dataclass(frozen=True)
class Joint:
    name: str  # URDF joint name
    servo_id: int
    sign: int  # +1 if servo steps increase with the URDF joint angle, -1 if they decrease
    group: str

    def to_radians(self, steps: int) -> float:
        return self.sign * (steps - ZERO_STEPS) * RADIANS_PER_STEP

    def to_steps(self, radians: float) -> int:
        return ZERO_STEPS + self.sign * round(radians / RADIANS_PER_STEP)


@dataclass(frozen=True)
class BusConfig:
    name: str
    port: str
    baudrate: int
    joints: tuple[Joint, ...]
    groups: dict[str, str]  # group name -> action
    poses: dict[str, dict[str, float]]  # pose name -> joint name -> radians

    def joint(self, name: str) -> Joint:
        return next(j for j in self.joints if j.name == name)

    def group(self, name: str) -> tuple[Joint, ...]:
        return tuple(j for j in self.joints if j.group == name)

    @property
    def servo_ids(self) -> list[int]:
        return [j.servo_id for j in self.joints]


def load_bus_config(path: Path) -> BusConfig:
    raw = yaml.safe_load(path.read_text())
    joints = tuple(
        Joint(name, servo_id=int(spec['id']), sign=int(spec['sign']), group=str(spec['group']))
        for name, spec in raw['joints'].items()
    )
    groups = {str(group): str(action) for group, action in raw['groups'].items()}
    poses = {
        pose: {joint: float(radians) for joint, radians in positions.items()}
        for pose, positions in raw.get('poses', {}).items()
    }

    ids = [j.servo_id for j in joints]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{path}: servo IDs must be unique, got {ids}')
    for j in joints:
        if j.sign not in (1, -1):
            raise ValueError(f'{path}: {j.name} sign must be 1 or -1, got {j.sign}')
        if j.group not in groups:
            raise ValueError(f'{path}: {j.name} is in group {j.group}, which is not under groups')
    for group, action in groups.items():
        if action not in ACTIONS:
            raise ValueError(f'{path}: group {group} has action {action}, not one of {ACTIONS}')
        if action == 'gripper_command' and sum(j.group == group for j in joints) != 1:
            raise ValueError(f'{path}: gripper group {group} must have exactly one joint')
    names = {j.name for j in joints}
    for pose, positions in poses.items():
        if unknown := positions.keys() - names:
            raise ValueError(f'{path}: pose {pose} names unknown joints {sorted(unknown)}')

    return BusConfig(
        name=path.stem,
        port=str(raw['port']),
        baudrate=int(raw['baudrate']),
        joints=joints,
        groups=groups,
        poses=poses,
    )
