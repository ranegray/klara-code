"""The agent: Claude writing and running code on the robot until a task is done.

Claude has one tool, run_code. The robot API part of its system prompt is generated from the API's
own docstrings and source, so the manual can't drift from the code. Every run waits for the user's
approval. Everything that happens is appended to a session log, one JSON object per line.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime
import inspect
import json
from pathlib import Path
import textwrap

import anthropic

from klara.robot import (
    Blocked, GripResult, MoveResult, Robot, RobotError, ToolPose, Unreachable, Unsafe)
from klara.runner import AGENT_CALLS, DEFAULT_TIMEOUT, MAX_TIMEOUT, CodeRunner, RunResult

DEFAULT_MODEL = 'claude-opus-5-5'
MAX_TOKENS = 32_000  # per response
MAX_RUNS = 30  # run_code calls per task
JSON_RETRIES = 2  # times to re-ask when a tool call arrives as JSON that can't be parsed

DIM, BOLD, RESET = '\033[2m', '\033[1m', '\033[0m'

RUN_CODE = {
    'name': 'run_code',
    'description': (
        'Run Python on the robot, in a long-lived process where robot and your earlier variables '
        'are defined. Blocks until the code finishes. Returns what it printed, the value of its '
        'last line, and any error.'),
    'eager_input_streaming': True,
    'input_schema': {
        'type': 'object',
        'properties': {
            'code': {'type': 'string', 'description': 'The Python to run.'},
            'timeout': {
                'type': 'number',
                'description': f'Seconds before the run is interrupted. Default '
                               f'{DEFAULT_TIMEOUT:g}, at most {MAX_TIMEOUT:g}.'},
        },
        'required': ['code'],
    },
}

PROMPT = """\
You control a real robot arm by writing Python that runs on the robot. You're working with the \
person beside it, who gives you tasks.

# The robot
An XLeRobot: a wheeled cart with a 5-joint arm and a two-finger gripper (its left arm) on top. \
You have no camera yet, so you can't see; the user tells you where things are.

Positions are meters in the robot's base frame: x forward, y left, z up, from the floor under the \
cart's center. The cart is a box from x -0.19 to 0.20 and y -0.23 to 0.23, up to its top at z \
0.775. The arm stands on the top near the front-left corner. Its shoulder is at about (0.12, \
0.16, 0.89), and the gripper reaches about 0.35 m from there. Each task starts with the gripper \
at (0.30, 0.17, 0.85), pointing 60 degrees down.

With 5 joints, the arm controls the gripper's position, its pitch and its roll. Its heading is \
not free: the gripper always points away from the arm's base, seen from above. Some combinations \
of position, pitch and roll are out of reach, especially near the edge of the workspace.

# Running code
run_code runs Python in one long-lived process, like a notebook: variables, functions and imports \
persist from one run to the next. Already defined: robot, np, math, and the exceptions below. \
Robot calls block until the robot has finished. Print what you want to see; the value of the last \
line, if it's an expression, is returned too.

Each run waits for the user's approval, and they may decline and tell you why. A run is \
interrupted after its timeout ({default_timeout:g} s unless you ask for more, at most \
{max_timeout:g} s). If a run had to be killed, the process restarts and earlier variables are \
gone; the result says so. A task may use at most {max_runs} runs.

# The robot API
{api}

# How to work
- Work in steps you can check: move, read the result, then continue.
- Results report where the gripper actually ended up. Trust them over your plan.
- When the task is done, or can't be done, stop calling run_code and say briefly what happened.
"""


def system_prompt() -> str:
    return PROMPT.format(api=_manual(), default_timeout=DEFAULT_TIMEOUT,
                         max_timeout=MAX_TIMEOUT, max_runs=MAX_RUNS)


def _manual() -> str:
    """The robot API as the agent's code sees it, from the code itself."""
    parts = []
    for name in AGENT_CALLS:
        method = getattr(Robot, name)
        signature = inspect.signature(method)
        params = ', '.join(str(p.replace(annotation=inspect.Parameter.empty))
                           for p in list(signature.parameters.values())[1:])  # without self
        doc = textwrap.indent(inspect.getdoc(method), '    ')
        parts.append(f'robot.{name}({params}) -> {signature.return_annotation}\n{doc}')
    for cls in (ToolPose, MoveResult, GripResult, RobotError, Unreachable, Unsafe, Blocked):
        parts.append(inspect.getsource(cls).rstrip())
    return '\n\n'.join(parts)


class Agent:
    """One session with Claude: a task, then any follow-ups, sharing one conversation."""

    def __init__(self, runner: CodeRunner, log: Path, model: str = DEFAULT_MODEL) -> None:
        self._client = anthropic.Anthropic()
        self._runner = runner
        self._model = model
        self._system = system_prompt()
        self._messages: list[dict] = []
        self._log = log.open('a')
        self._record('session', model=model, system=self._system, tools=[RUN_CODE])

    def close(self) -> None:
        self._log.close()

    def ask(self, task: str) -> None:
        """Work on a task until Claude stops calling run_code. Ctrl-C stops it early."""
        self._add_user([{'type': 'text', 'text': task}])
        self._record('user', text=task)
        self._runs, self._approve_all, self._usage = 0, False, Counter()
        try:
            self._work()
        except KeyboardInterrupt:  # while Claude was responding: its partial reply is dropped
            print(f'\n{BOLD}[interrupted]{RESET}')
            self._record('interrupted')
        u = self._usage
        tokens_in = (u['input_tokens'] + u['cache_creation_input_tokens']
                     + u['cache_read_input_tokens'])
        print(f'{DIM}[{self._runs} runs; tokens: {tokens_in} in '
              f'({u["cache_read_input_tokens"]} read from cache), {u["output_tokens"]} out]{RESET}')

    def _work(self) -> None:
        while True:
            message = self._respond()
            if message is None:
                return
            self._messages.append({'role': 'assistant', 'content': message.content})
            calls = [b for b in message.content if b.type == 'tool_use']
            if not calls or message.stop_reason == 'refusal':
                return
            results, stop = [], False
            for call in calls:
                if stop:
                    results.append(_tool_result(call.id, 'Not run: the task was stopped.', True))
                    continue
                try:
                    result, stop = self._handle(call, message.stop_reason)
                except KeyboardInterrupt:
                    print(f'\n{BOLD}[interrupted: robot stopped]{RESET}')
                    self._record('interrupted')
                    result, stop = _tool_result(
                        call.id, 'The user pressed Ctrl-C. If the code was running, it and the '
                                 'robot were stopped.', True), True
                results.append(result)
            self._add_user(results)
            if stop:
                return

    def _respond(self):
        """Stream Claude's next message to the terminal and return it, or None if the API failed."""
        for attempt in range(JSON_RETRIES + 1):
            try:
                with self._client.messages.stream(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=self._system,
                    tools=[RUN_CODE],
                    tool_choice={'type': 'auto', 'disable_parallel_tool_use': True},
                    thinking={'type': 'adaptive', 'display': 'summarized'},
                    cache_control={'type': 'ephemeral'},  # caches the conversation so far
                    messages=self._messages,
                ) as stream:
                    _show(stream)
                    message = stream.get_final_message()
                break
            except ValueError:  # a tool call whose JSON couldn't be parsed at all; ask again
                if attempt == JSON_RETRIES:
                    raise
            except anthropic.APIError as e:
                print(f'\n{BOLD}Claude API error: {e}{RESET}')
                self._record('api_error', error=str(e))
                return None
        usage = message.usage.model_dump()
        self._usage.update({k: v for k, v in usage.items() if isinstance(v, int)})
        self._record('assistant', content=[b.model_dump() for b in message.content],
                     stop_reason=message.stop_reason, usage=usage)
        return message

    def _handle(self, call, stop_reason: str) -> tuple[dict, bool]:
        """Run one run_code call, if it's whole, allowed and approved. Returns (result, stop)."""
        args = call.input
        code = args.get('code') if isinstance(args, dict) else None
        timeout = args.get('timeout') if isinstance(args, dict) else None
        if stop_reason == 'max_tokens':
            return _tool_result(call.id, 'Not run: your response reached its length limit before '
                                         'the code was complete. Send shorter code.', True), False
        if not isinstance(code, str) or not (timeout is None or isinstance(timeout, (int, float))):
            return _tool_result(call.id, 'Not run: run_code takes code (a string) and optionally '
                                         'timeout (seconds).', True), False
        if self._runs >= MAX_RUNS:
            return _tool_result(call.id, f'Not run: this task has used all {MAX_RUNS} runs.',
                                True), True

        heading = 'run_code' + (f' (timeout {timeout:g} s)' if timeout else '')
        print(f'\n{BOLD}── {heading} ──{RESET}')
        for line in code.splitlines():
            print(f'{DIM}│{RESET} {line}')
        approved, reason = self._approve()
        self._record('approval', approved=approved, reason=reason)
        if not approved:
            text = 'The user declined to run this code.' + (f' They said: {reason}' if reason
                                                             else '')
            return _tool_result(call.id, text, False), False

        self._runs += 1
        result = self._runner.run(code, timeout)
        self._record('run', code=code, timeout=timeout, **asdict(result))
        if result.value:
            print(result.value)
        if result.error:
            print(result.error.rstrip())
        print(f'{DIM}[{_status(result)}]{RESET}')
        return _tool_result(call.id, _describe(result), bool(result.error or result.stopped)), False

    def _approve(self) -> tuple[bool, str]:
        if self._approve_all:
            return True, ''
        while True:
            answer = input('Run it? [y]es, [n]o, [a]ll for this task: ').strip().lower()
            if answer in ('y', 'yes'):
                return True, ''
            if answer in ('a', 'all'):
                self._approve_all = True
                return True, ''
            if answer in ('n', 'no'):
                return False, input('Why? (optional; Claude sees this) ').strip()

    def _add_user(self, content: list[dict]) -> None:
        # A follow-up after a stopped task lands next to the last tool results, in one turn.
        if self._messages and self._messages[-1]['role'] == 'user':
            self._messages[-1]['content'].extend(content)
        else:
            self._messages.append({'role': 'user', 'content': content})

    def _record(self, kind: str, **fields) -> None:
        entry = {'time': datetime.now().isoformat(timespec='milliseconds'), 'kind': kind, **fields}
        self._log.write(json.dumps(entry, default=str) + '\n')
        self._log.flush()


def _show(stream) -> None:
    """Print Claude's thinking (dimmed) and text as they arrive."""
    for event in stream:
        if event.type == 'content_block_start':
            if event.content_block.type == 'thinking':
                print(DIM, end='')
            elif event.content_block.type == 'tool_use':
                print(f'{DIM}[writing code]', end='', flush=True)
        elif event.type == 'content_block_delta':
            if event.delta.type == 'thinking_delta':
                print(event.delta.thinking, end='', flush=True)
            elif event.delta.type == 'text_delta':
                print(event.delta.text, end='', flush=True)
        elif event.type == 'content_block_stop':
            print(RESET)


def _describe(result: RunResult) -> str:
    """A run's result as Claude reads it."""
    parts = []
    if result.output:
        parts.append(f'Output:\n{result.output}')
    if result.value:
        parts.append(f'Value: {result.value}')
    if result.error:
        parts.append(f'Error:\n{result.error}')
    parts.append(_status(result) + '.')
    return '\n\n'.join(parts)


def _status(result: RunResult) -> str:
    status = f'took {result.seconds:.1f} s'
    if result.stopped:
        status += f'; stopped: {result.stopped}'
    if result.restarted:
        status += '; the process was restarted, so variables from earlier runs are gone'
    return status


def _tool_result(tool_use_id: str, text: str, error: bool) -> dict:
    return {'type': 'tool_result', 'tool_use_id': tool_use_id, 'content': text, 'is_error': error}
