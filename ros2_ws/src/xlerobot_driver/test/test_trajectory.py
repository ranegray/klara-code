import pytest

from xlerobot_driver.trajectory import Trajectory


def test_interpolates_between_waypoints_on_time():
    traj = Trajectory.through((0.0, 0.0), [1.0, 3.0], [(1.0, -1.0), (1.0, 1.0)])
    assert traj.sample(0.5) == pytest.approx((0.5, -0.5))
    assert traj.sample(1.0) == pytest.approx((1.0, -1.0))
    assert traj.sample(2.0) == pytest.approx((1.0, 0.0))


def test_holds_the_ends():
    traj = Trajectory.through((0.0,), [2.0], [(4.0,)])
    assert traj.sample(-1.0) == (0.0,)
    assert traj.sample(10.0) == (4.0,)
    assert traj.duration == 2.0


def test_max_speed_includes_the_move_from_start():
    traj = Trajectory.through((0.0, 0.0), [0.5, 2.5], [(1.0, 0.0), (1.0, 1.0)])
    assert traj.max_speed() == pytest.approx(2.0)


@pytest.mark.parametrize('times', [[0.0], [1.0, 1.0], [2.0, 1.0], []])
def test_rejects_bad_times(times):
    with pytest.raises(ValueError):
        Trajectory.through((0.0,), times, [(1.0,)] * len(times))


def test_rejects_wrong_width():
    with pytest.raises(ValueError):
        Trajectory.through((0.0, 0.0), [1.0], [(1.0,)])
