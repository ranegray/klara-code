import math

import pytest

from xlerobot_driver.bus_config import Joint


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('zero_steps', [2048, 3072])
def test_steps_and_radians_round_trip(sign, zero_steps):
    joint = Joint('j', servo_id=1, sign=sign, group='g', zero_steps=zero_steps)
    assert joint.to_radians(zero_steps) == 0.0
    for steps in (0, 1000, 4095):
        assert joint.to_steps(joint.to_radians(steps)) == steps


def test_pitch_range_fits_in_one_turn():
    pitch = Joint('Pitch_L', servo_id=2, sign=-1, group='left_arm', zero_steps=3072)
    low, high = pitch.to_steps(3.4), pitch.to_steps(-0.22)
    assert 0 < low < high < 4095
    assert pitch.to_radians(3072 - 1024) == pytest.approx(math.pi / 2)
