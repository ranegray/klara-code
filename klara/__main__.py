"""klara: give the robot a task in plain words.

  docker compose exec robot klara [--model MODEL] [TASK ...]

Claude writes code for the task and runs it on the robot, asking before each run. When it's done
you can follow up in the same session; an empty line ends it. Ctrl-C stops the robot.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

import rclpy
from rclpy.signals import SignalHandlerOptions

from klara.agent import Agent, DEFAULT_MODEL, DIM, RESET
from klara.robot import Robot, RobotError
from klara.runner import CodeRunner

RUNS = Path(__file__).resolve().parents[1] / 'runs'  # session logs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='klara', description=__doc__.splitlines()[0])
    parser.add_argument('--model', default=DEFAULT_MODEL, help=f'default {DEFAULT_MODEL}')
    parser.add_argument('task', nargs='*', help='what to do; asked for if left out')
    args = parser.parse_args(argv)
    if not os.environ.get('ANTHROPIC_API_KEY'):
        print('error: ANTHROPIC_API_KEY is not set. Put it in .env next to compose.yaml.',
              file=sys.stderr)
        return 1
    task = ' '.join(args.task) or _prompt()
    if not task:
        return 0

    log = RUNS / f'{datetime.now():%Y-%m-%d_%H-%M-%S}.jsonl'
    # Without ROS's signal handling, Ctrl-C raises KeyboardInterrupt and ROS stays up to stop
    # the robot.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    try:
        with Robot() as robot:
            print('Moving to the ready pose.')
            robot.move_joints(robot.poses['ready'])
        with CodeRunner(on_output=_show_output) as runner:
            agent = Agent(runner, log, args.model)
            try:
                while task:
                    agent.ask(task)
                    task = _prompt()
            finally:
                agent.close()
        print(f'Session log: runs/{log.name}')
    except RobotError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\nStopped.')
        return 130
    finally:
        rclpy.shutdown()
    return 0


def _prompt() -> str:
    try:
        return input('\n> ').strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ''


def _show_output(text: str) -> None:
    sys.stdout.write(f'{DIM}{text}{RESET}')
    sys.stdout.flush()


if __name__ == '__main__':
    sys.exit(main())
