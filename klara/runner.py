"""Run the agent's code in a long-lived worker process, like a notebook kernel.

Variables persist from one run to the next. A run that goes past its timeout is interrupted; if
the code won't stop, the worker is killed and replaced, losing its variables. Whenever a run ends
that way, every arm and gripper motion is cancelled, so stopped code can't leave the robot moving.

The code gets one object, robot, offering only the calls in the agent's manual.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import linecache
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
import traceback
from typing import Callable

from action_msgs.srv import CancelGoal
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from klara.robot import (
    ARM_ACTION, Blocked, GRIPPER_ACTION, Robot, RobotError, Unreachable, Unsafe)

DEFAULT_TIMEOUT = 120.0  # s per run
MAX_TIMEOUT = 300.0  # s, the longest a run may ask for
INTERRUPT_GRACE = 3.0  # s for interrupted code to stop before the worker is killed
START_TIMEOUT = 30.0  # s for a new worker to connect to the robot
OUTPUT_LIMIT = 10_000  # characters of output kept, from the end

# What the agent's code can call on robot.
AGENT_CALLS = ('move_to', 'open_gripper', 'close_gripper', 'get_pose')


@dataclass(frozen=True)
class RunResult:
    output: str  # what the code printed, cut to the last OUTPUT_LIMIT characters
    value: str | None  # repr of the last line, if it was an expression with a value
    error: str | None  # traceback through the agent's code, if it raised
    seconds: float
    stopped: str | None  # why the run was cut short, if it was
    restarted: bool  # the worker was replaced, so variables from earlier runs are gone


class CodeRunner:
    """Runs code in the worker. on_output gets the code's printed output as it's printed."""

    def __init__(self, on_output: Callable[[str], None] | None = None) -> None:
        self._on_output = on_output or (lambda _: None)
        # Without ROS's own signal handling, Ctrl-C raises KeyboardInterrupt instead of shutting
        # ROS down, so the robot can still be stopped afterwards.
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self._node = rclpy.create_node('klara_runner')
        self._executor = SingleThreadedExecutor()  # its own, to shut down before the node goes
        self._executor.add_node(self._node)
        self._cancels = [self._node.create_client(CancelGoal, f'{action}/_action/cancel_goal')
                         for action in (ARM_ACTION, GRIPPER_ACTION)]
        # A fresh interpreter, rather than a copy of this one with its ROS connections.
        self._context = multiprocessing.get_context('spawn')
        self._process = None
        self._start()

    def run(self, code: str, timeout: float | None = None) -> RunResult:
        """Run code in the worker and wait for it to finish, at most timeout seconds."""
        if not self._process.is_alive():
            self._start()
        timeout = min(timeout or DEFAULT_TIMEOUT, MAX_TIMEOUT)
        started = time.monotonic()
        output: list[str] = []
        stopped = None
        try:
            self._conn.send(code)
            done = self._collect(output, started + timeout)
            if done is None:
                stopped = f'timed out after {timeout:g} s'
                done = self._interrupt(output)
        except KeyboardInterrupt:
            # Ctrl-C here: stop the run the same way, then pass the Ctrl-C on.
            if self._interrupt(output) is None:
                self._process.kill()
                self._process.join()
            self.stop_robot()
            raise
        except (EOFError, OSError):
            self._process.join(1.0)
            done, stopped = None, f'the worker died (exit code {self._process.exitcode})'

        if done is None:  # it never stopped, or it died; the next run starts a new worker
            self._process.kill()
            self._process.join()
        if stopped:  # after the kill, so the code can't start another motion
            self.stop_robot()
        return RunResult(
            output=_tail(''.join(output)),
            value=done and done['value'] and _tail(done['value']),
            error=done and done['error'],
            seconds=time.monotonic() - started,
            stopped=stopped,
            restarted=done is None)

    def stop_robot(self) -> None:
        """Cancel every arm and gripper motion, whoever started it."""
        for client in self._cancels:
            if client.wait_for_service(timeout_sec=1.0):
                # An empty request, with no goal ID and no time, cancels every goal.
                future = client.call_async(CancelGoal.Request())
                self._executor.spin_until_future_complete(future, timeout_sec=2.0)

    def close(self) -> None:
        if self._process.is_alive():
            self._conn.send(None)
            self._process.join(5.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
        self._executor.shutdown()
        self._node.destroy_node()
        if self._owns_rclpy:
            rclpy.shutdown()

    def __enter__(self) -> CodeRunner:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _start(self) -> None:
        self._conn, child = self._context.Pipe()
        self._process = self._context.Process(target=_serve, args=(child,), name='klara-worker',
                                              daemon=True)
        self._process.start()
        child.close()
        try:
            if not self._conn.poll(START_TIMEOUT):
                raise RobotError(f'the worker did not connect to the robot in {START_TIMEOUT:g} s')
            kind, detail = self._conn.recv()
            if kind != 'ready':
                raise RobotError(f'the worker could not connect to the robot: {detail}')
        except (RobotError, EOFError) as e:
            self._process.kill()
            self._process.join()
            raise RobotError(str(e) or 'the worker died while starting') from e

    def _interrupt(self, output: list[str]) -> dict | None:
        """Raise KeyboardInterrupt in the running code, and give it a moment to stop."""
        os.kill(self._process.pid, signal.SIGINT)
        return self._collect(output, time.monotonic() + INTERRUPT_GRACE)

    def _collect(self, output: list[str], deadline: float) -> dict | None:
        """Gather what the worker sends until its run is done (its result) or the deadline (None)."""
        while (remaining := deadline - time.monotonic()) > 0:
            if not self._conn.poll(remaining):
                return None
            kind, payload = self._conn.recv()
            if kind == 'output':
                output.append(payload)
                self._on_output(payload)
            else:
                return payload
        return None


class RobotAPI:
    """The robot as the agent's code gets it: only the calls in its manual."""

    def __init__(self, robot: Robot) -> None:
        for name in AGENT_CALLS:
            setattr(self, name, getattr(robot, name))

    def __repr__(self) -> str:
        return f'robot with {", ".join(f"{name}()" for name in AGENT_CALLS)}'


# The worker process.

def _serve(conn) -> None:
    """Connect to the robot, then run the code the parent sends until it sends None."""
    # The parent interrupts a run with SIGINT. Without ROS's signal handling, that raises
    # KeyboardInterrupt in the code (and the robot cancels its motion) instead of shutting ROS down.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # Its own process group, so Ctrl-C at the terminal reaches only the parent, which then
    # interrupts the code itself. One path for every stop.
    os.setpgrp()
    os.environ.pop('ANTHROPIC_API_KEY', None)  # the agent's code has no use for it
    try:
        robot = Robot()
    except Exception as e:
        conn.send(('failed', str(e)))
        return
    namespace = {
        'robot': RobotAPI(robot), 'np': np, 'math': math,
        'RobotError': RobotError, 'Unreachable': Unreachable, 'Unsafe': Unsafe, 'Blocked': Blocked,
    }
    sys.stdout = sys.stderr = _Output(conn)
    conn.send(('ready', None))
    runs = 0
    while True:
        try:
            code = conn.recv()
        except KeyboardInterrupt:  # an interrupt that arrived just after a run finished
            continue
        except EOFError:
            break
        if code is None:
            break
        runs += 1
        conn.send(('done', _execute(code, f'<run {runs}>', namespace)))
    robot.close()
    rclpy.shutdown()


def _execute(code: str, filename: str, namespace: dict) -> dict:
    """Run code like a notebook cell: if the last line is an expression, return its value."""
    linecache.cache[filename] = (len(code), None, code.splitlines(keepends=True), filename)
    value = error = None
    try:
        tree = ast.parse(code, filename)
        last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, filename, 'exec'), namespace)
        if last is not None:
            result = eval(compile(ast.Expression(last.value), filename, 'eval'), namespace)
            if result is not None:
                value = repr(result)
    except BaseException as e:  # everything, including the KeyboardInterrupt of a timeout
        error = _traceback(e)
    return {'value': value, 'error': error}


def _traceback(e: BaseException) -> str:
    """The traceback through the agent's own code only, not through klara's."""
    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename.startswith('<run ')]
    lines = traceback.format_exception_only(e)
    if frames:
        lines = ['Traceback (most recent call last):\n', *traceback.format_list(frames), *lines]
    return ''.join(lines)


class _Output:
    """Stands in for stdout and stderr in the worker, sending what's written to the parent."""

    encoding = 'utf-8'

    def __init__(self, conn) -> None:
        self._conn = conn
        self._lock = threading.Lock()  # other threads (ROS callbacks) may print too

    def write(self, text: str) -> int:
        if text:
            with self._lock:
                self._conn.send(('output', text))
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _tail(text: str) -> str:
    if len(text) <= OUTPUT_LIMIT:
        return text
    return f'[... {len(text) - OUTPUT_LIMIT} earlier characters cut ...]\n' + text[-OUTPUT_LIMIT:]
