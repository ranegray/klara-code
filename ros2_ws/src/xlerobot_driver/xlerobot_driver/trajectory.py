"""Joint trajectories: timed waypoints, sampled by linear interpolation. No ROS imports."""

from __future__ import annotations

import bisect
from dataclasses import dataclass


@dataclass(frozen=True)
class Trajectory:
    times: tuple[float, ...]  # seconds from start, strictly increasing, the first one 0
    positions: tuple[tuple[float, ...], ...]  # one row per time, one column per joint

    @classmethod
    def through(
        cls, start: tuple[float, ...], times: list[float], positions: list[tuple[float, ...]]
    ) -> Trajectory:
        """Build a trajectory that begins at `start` and passes through each waypoint on time."""
        if not times:
            raise ValueError('trajectory has no points')
        if any(len(row) != len(start) for row in positions):
            raise ValueError(f'every point needs {len(start)} positions')
        all_times = [0.0, *times]
        if any(t1 <= t0 for t0, t1 in zip(all_times, all_times[1:])):
            raise ValueError('time_from_start must be above 0 and strictly increasing '
                             '(the trajectory starts from the current position at time 0)')
        return cls(tuple(all_times), (tuple(start), *map(tuple, positions)))

    @property
    def duration(self) -> float:
        return self.times[-1]

    def sample(self, t: float) -> tuple[float, ...]:
        if t >= self.duration:
            return self.positions[-1]
        i = bisect.bisect_right(self.times, t)  # times[i - 1] <= t < times[i]
        if i == 0:
            return self.positions[0]
        t0, t1 = self.times[i - 1], self.times[i]
        a = (t - t0) / (t1 - t0)
        before, after = self.positions[i - 1], self.positions[i]
        return tuple(p0 + a * (p1 - p0) for p0, p1 in zip(before, after))

    def max_speed(self) -> float:
        """Fastest any joint moves in any segment, in position units per second."""
        segments = zip(self.times, self.times[1:], self.positions, self.positions[1:])
        return max(
            abs(p1 - p0) / (t1 - t0)
            for t0, t1, row0, row1 in segments
            for p0, p1 in zip(row0, row1)
        )
