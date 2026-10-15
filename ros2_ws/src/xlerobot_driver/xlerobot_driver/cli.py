"""servo_bus: inspect and calibrate a servo bus from the terminal. Runs without ROS.

  servo_bus dump                     live state and the calibration stored on each servo
  servo_bus torque-off [JOINT ...]   turn torque off, on every joint unless some are named
  servo_bus calibrate JOINT ...      store a new zero and range on these joints' servos
"""

from __future__ import annotations

import argparse
from pathlib import Path
import select
import sys

from xlerobot_driver.bus_config import BusConfig, Joint, load_bus_config, ZERO_STEPS
from xlerobot_driver.servo_bus import (
    BusError, Calibration, ServoBus, ServoState, STEPS_PER_TURN, TORQUE_ENABLE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='servo_bus', description=__doc__.splitlines()[0])
    parser.add_argument('--config', type=Path, default=_default_config(), help='bus config YAML')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('dump', help='print live state and stored calibration')
    commands.add_parser('torque-off', help='turn torque off').add_argument(
        'joints', nargs='*', metavar='JOINT')
    commands.add_parser('calibrate', help='store a new zero and range').add_argument(
        'joints', nargs='+', metavar='JOINT')
    args = parser.parse_args(argv)

    config = load_bus_config(args.config)
    names = {j.name for j in config.joints}
    selected = getattr(args, 'joints', None) or []
    if unknown := set(selected) - names:
        parser.error(f'unknown joints {sorted(unknown)}; {config.name} has {sorted(names)}')
    joints = tuple(j for j in config.joints if j.name in selected) if selected else config.joints

    try:
        with ServoBus(config.port, config.baudrate) as bus:
            if args.command == 'dump':
                dump(bus, config)
            elif args.command == 'torque-off':
                return torque_off(bus, joints)
            else:
                calibrate(bus, joints)
    except BusError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    return 0


def dump(bus: ServoBus, config: BusConfig) -> None:
    print(f'{"joint":16} {"id":>2} {"group":12} {"steps":>5} {"rad":>7} {"torque":>6} '
          f'{"offset":>6} {"min":>4} {"max":>4} {"volts":>5} {"temp":>4}  faults')
    states = bus.read_state(config.servo_ids)
    for j in config.joints:
        state = states[j.servo_id]
        if state is None:
            print(f'{j.name:16} {j.servo_id:2} {j.group:12} no reply')
            continue
        cal = bus.read_calibration(j.servo_id)
        torque = 'on' if bus.read(j.servo_id, TORQUE_ENABLE) else 'off'
        print(f'{j.name:16} {j.servo_id:2} {j.group:12} {state.position:5} '
              f'{j.to_radians(state.position):+7.3f} {torque:>6} {cal.homing_offset:6} '
              f'{cal.min_position:4} {cal.max_position:4} {state.voltage:5.1f} '
              f'{state.temperature:4}  '
              f'{", ".join(state.faults) or "-"}')


def torque_off(bus: ServoBus, joints: tuple[Joint, ...]) -> int:
    # Try every joint even if one fails, so a single dead servo can't leave the rest powered.
    failed = []
    for j in joints:
        try:
            bus.set_torque([j.servo_id], False)
        except BusError as e:
            failed.append(j.name)
            print(f'error: {e}', file=sys.stderr)
    print('torque off:', ', '.join(j.name for j in joints if j.name not in failed) or 'none')
    return 1 if failed else 0


def calibrate(bus: ServoBus, joints: tuple[Joint, ...]) -> None:
    ids = [j.servo_id for j in joints]
    bus.set_torque(ids, False)
    input(f'Torque is off on {", ".join(j.name for j in joints)}.\n'
          'Hold them at the URDF zero pose, then press Enter.')

    # Present position = raw position - homing offset (mod one turn). Choose the offset that
    # makes the current raw position read ZERO_STEPS, and open the range fully until the sweep
    # records it.
    offsets = {}
    states = _read(bus, ids)
    for j in joints:
        old_offset = bus.read_calibration(j.servo_id).homing_offset
        raw = (states[j.servo_id].position + old_offset) % STEPS_PER_TURN
        offsets[j.servo_id] = raw - ZERO_STEPS
        bus.write_calibration(j.servo_id, Calibration(offsets[j.servo_id], 0, STEPS_PER_TURN - 1))
    states = _read(bus, ids)
    for j in joints:
        print(f'  {j.name}: zero stored, now reads {states[j.servo_id].position} '
              f'(target {ZERO_STEPS})')

    print('Move each joint slowly through its full range, then press Enter.')
    low, high = _sweep(bus, joints)
    for j in joints:
        lo, hi = low[j.servo_id], high[j.servo_id]
        bus.write_calibration(j.servo_id, Calibration(offsets[j.servo_id], lo, hi))
        print(f'  {j.name}: range {lo}..{hi} steps, '
              f'{j.to_radians(lo):+.3f}..{j.to_radians(hi):+.3f} rad')


def _sweep(bus: ServoBus, joints: tuple[Joint, ...]) -> tuple[dict[int, int], dict[int, int]]:
    """Track each joint's lowest and highest position until Enter is pressed."""
    ids = [j.servo_id for j in joints]
    last = {i: s.position for i, s in _read(bus, ids).items()}
    low, high = dict(last), dict(last)
    while not select.select([sys.stdin], [], [], 0.02)[0]:
        for servo_id, state in _read(bus, ids).items():
            if abs(state.position - last[servo_id]) > STEPS_PER_TURN // 2:
                raise BusError(f'servo {servo_id} wrapped past 0/{STEPS_PER_TURN - 1}: it went '
                               'more than half a turn from zero, which one-turn position '
                               'readings cannot represent. Sweep only the range the servo '
                               'should allow. The range was not stored.')
            last[servo_id] = state.position
            low[servo_id] = min(low[servo_id], state.position)
            high[servo_id] = max(high[servo_id], state.position)
        ranges = (f'{j.name} {low[j.servo_id]}..{high[j.servo_id]}' for j in joints)
        print('\r  ' + '   '.join(ranges), end='', flush=True)
    sys.stdin.readline()
    print()
    return low, high


def _read(bus: ServoBus, ids: list[int]) -> dict[int, ServoState]:
    states = bus.read_state(ids)
    if silent := [i for i, s in states.items() if s is None]:
        raise BusError(f'no reply from servos {silent}')
    return states


def _default_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('xlerobot_driver')) / 'config' / 'bus1.yaml'
    except (ImportError, LookupError):  # running from the source tree, or without ROS
        return Path(__file__).resolve().parents[1] / 'config' / 'bus1.yaml'


if __name__ == '__main__':
    sys.exit(main())
