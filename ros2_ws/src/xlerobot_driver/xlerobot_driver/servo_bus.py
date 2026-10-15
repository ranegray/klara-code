"""Feetech STS3215 servo bus: the control table and a small read/write layer over the Feetech SDK.

Values here are raw servo units (position steps, volts, degrees C). Joint names and radians live in
bus_config. Nothing here imports ROS, so the driver node and the servo_bus CLI can share it.
"""

from __future__ import annotations

from dataclasses import dataclass

import scservo_sdk as scs


@dataclass(frozen=True)
class Register:
    """One entry in the servo's control table."""

    address: int
    size: int  # bytes
    sign_bit: int | None = None  # Feetech stores signed values as sign-magnitude
    eeprom: bool = False  # survives power cycles; writable only while LOCK is 0


# The subset of the STS3215 control table we use. Everything from TORQUE_ENABLE on is RAM and
# resets at power-on.
MIN_POSITION_LIMIT = Register(9, 2, eeprom=True)
MAX_POSITION_LIMIT = Register(11, 2, eeprom=True)
HOMING_OFFSET = Register(31, 2, sign_bit=11, eeprom=True)  # present = raw position - offset
TORQUE_ENABLE = Register(40, 1)
ACCELERATION = Register(41, 1)
GOAL_POSITION = Register(42, 2)
LOCK = Register(55, 1)
PRESENT_POSITION = Register(56, 2)
PRESENT_VELOCITY = Register(58, 2, sign_bit=15)
PRESENT_LOAD = Register(60, 2, sign_bit=10)
PRESENT_VOLTAGE = Register(62, 1)  # tenths of a volt
PRESENT_TEMPERATURE = Register(63, 1)  # degrees C

STS3215_MODEL = 777
STEPS_PER_TURN = 4096

# A write that changes an EEPROM value takes the servo up to ~41 ms to acknowledge (measured),
# past the SDK's usual ~34 ms reply timeout.
EEPROM_REPLY_TIMEOUT_MS = 100

# Bits of the error byte every servo reply carries.
FAULT_BITS = {1: 'voltage', 2: 'angle sensor', 4: 'overheat', 8: 'overcurrent', 32: 'overload'}


def decode(register: Register, raw: int) -> int:
    if register.sign_bit is not None and raw & (1 << register.sign_bit):
        return -(raw & ((1 << register.sign_bit) - 1))
    return raw


def encode(register: Register, value: int) -> int:
    magnitude_bits = register.sign_bit if register.sign_bit is not None else 8 * register.size
    if abs(value) >= 1 << magnitude_bits or (value < 0 and register.sign_bit is None):
        raise ValueError(f'{value} does not fit the register at address {register.address}')
    return -value | (1 << register.sign_bit) if value < 0 else value


def faults(error: int) -> tuple[str, ...]:
    return tuple(name for bit, name in FAULT_BITS.items() if error & bit)


@dataclass(frozen=True)
class ServoState:
    position: int  # steps
    velocity: int  # steps per second
    load: int  # tenths of a percent of max torque, signed by direction
    voltage: float  # volts
    temperature: int  # degrees C
    faults: tuple[str, ...]


@dataclass(frozen=True)
class Calibration:
    homing_offset: int  # steps
    min_position: int  # steps, enforced by the servo itself
    max_position: int


class BusError(RuntimeError):
    """The bus could not be opened, or a servo did not answer."""


class ServoBus:
    """One serial bus of STS3215 servos. Not thread-safe: give it a single owner."""

    # One sync read covers present position through present temperature.
    _STATE_START = PRESENT_POSITION.address
    _STATE_LENGTH = PRESENT_TEMPERATURE.address + PRESENT_TEMPERATURE.size - _STATE_START

    def __init__(self, port: str, baudrate: int) -> None:
        self._port = scs.PortHandler(port)
        self._packets = scs.PacketHandler(0)  # 0 selects little-endian, the STS series byte order
        self._baudrate = baudrate

    def open(self) -> None:
        try:
            self._port.setBaudRate(self._baudrate)  # opens the port
        except OSError as e:
            raise BusError(f'could not open {self._port.getPortName()}: {e}') from e

    def close(self) -> None:
        self._port.closePort()

    def __enter__(self) -> ServoBus:
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def ping(self, servo_id: int) -> int | None:
        """Return the servo's model number, or None if nothing answers at this ID."""
        model, result, _ = self._packets.ping(self._port, servo_id)
        return model if result == scs.COMM_SUCCESS else None

    def read(self, servo_id: int, register: Register) -> int:
        data, result, _ = self._packets.readTxRx(
            self._port, servo_id, register.address, register.size)
        self._check(result, f'servo {servo_id}, reading address {register.address}')
        return decode(register, int.from_bytes(bytes(data), 'little'))

    def write(self, servo_id: int, register: Register, value: int) -> None:
        data = list(encode(register, value).to_bytes(register.size, 'little'))
        if register.eeprom:
            # Wait out the slow EEPROM acknowledgement. Timing out early would leave the late
            # reply on the bus, to be mistaken for the reply to the next command.
            result = self._packets.writeTxOnly(
                self._port, servo_id, register.address, register.size, data)
            if result == scs.COMM_SUCCESS:
                self._port.setPacketTimeoutMillis(EEPROM_REPLY_TIMEOUT_MS)
                _, result, _ = self._packets.readRx(self._port, servo_id, 0)
        else:
            result, _ = self._packets.writeTxRx(
                self._port, servo_id, register.address, register.size, data)
        self._check(result, f'servo {servo_id}, writing address {register.address}')

    def read_state(self, servo_ids: list[int]) -> dict[int, ServoState | None]:
        """Read every servo's live state in one sync read. Silent servos map to None."""
        result = self._packets.syncReadTx(
            self._port, self._STATE_START, self._STATE_LENGTH, servo_ids, len(servo_ids))
        if result != scs.COMM_SUCCESS:
            raise BusError(f'sync read failed: {self._packets.getTxRxResult(result)}')
        states: dict[int, ServoState | None] = {}
        for servo_id in servo_ids:
            data, result, error = self._packets.readRx(self._port, servo_id, self._STATE_LENGTH)
            if result != scs.COMM_SUCCESS:
                break
            states[servo_id] = self._parse_state(data, error)
        # A missing reply throws off the rest of a sync read, so ask the remaining servos one at
        # a time. Only the servos that really are silent end up as None.
        for servo_id in servo_ids[len(states):]:
            data, result, error = self._packets.readTxRx(
                self._port, servo_id, self._STATE_START, self._STATE_LENGTH)
            ok = result == scs.COMM_SUCCESS
            states[servo_id] = self._parse_state(data, error) if ok else None
        return states

    def write_goal_positions(self, goals: dict[int, int]) -> None:
        """Send every servo its goal position in one sync write. Servos don't reply to these."""
        params = []
        for servo_id, steps in goals.items():
            raw = encode(GOAL_POSITION, steps)
            params += [servo_id, *raw.to_bytes(GOAL_POSITION.size, 'little')]
        result = self._packets.syncWriteTxOnly(
            self._port, GOAL_POSITION.address, GOAL_POSITION.size, params, len(params))
        if result != scs.COMM_SUCCESS:
            raise BusError(f'sync write failed: {self._packets.getTxRxResult(result)}')

    def set_torque(self, servo_ids: list[int], enabled: bool) -> None:
        for servo_id in servo_ids:
            self.write(servo_id, TORQUE_ENABLE, int(enabled))

    def read_calibration(self, servo_id: int) -> Calibration:
        return Calibration(
            homing_offset=self.read(servo_id, HOMING_OFFSET),
            min_position=self.read(servo_id, MIN_POSITION_LIMIT),
            max_position=self.read(servo_id, MAX_POSITION_LIMIT),
        )

    def write_calibration(self, servo_id: int, calibration: Calibration) -> None:
        """Store a calibration in EEPROM, then read it back to confirm. Torque must be off."""
        if self.read(servo_id, TORQUE_ENABLE):
            raise BusError(f'servo {servo_id}: torque must be off to write calibration')
        self.write(servo_id, LOCK, 0)
        try:
            self.write(servo_id, HOMING_OFFSET, calibration.homing_offset)
            self.write(servo_id, MIN_POSITION_LIMIT, calibration.min_position)
            self.write(servo_id, MAX_POSITION_LIMIT, calibration.max_position)
        finally:
            self.write(servo_id, LOCK, 1)
        stored = self.read_calibration(servo_id)
        if stored != calibration:
            raise BusError(f'servo {servo_id}: wrote {calibration} but read back {stored}')

    def _parse_state(self, data: list[int], error: int) -> ServoState:
        def field(register: Register) -> int:
            offset = register.address - self._STATE_START
            raw = int.from_bytes(bytes(data[offset:offset + register.size]), 'little')
            return decode(register, raw)

        return ServoState(
            position=field(PRESENT_POSITION),
            velocity=field(PRESENT_VELOCITY),
            load=field(PRESENT_LOAD),
            voltage=field(PRESENT_VOLTAGE) / 10,
            temperature=field(PRESENT_TEMPERATURE),
            faults=faults(error),
        )

    def _check(self, result: int, what: str) -> None:
        if result != scs.COMM_SUCCESS:
            raise BusError(f'{what}: {self._packets.getTxRxResult(result)}')
