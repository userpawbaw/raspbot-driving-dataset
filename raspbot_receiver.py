import argparse
import importlib
import os
import re
import tempfile
import signal
import socket
import sys
import time
from collections import deque
from threading import Event, Lock, Thread
from pathlib import Path
from uuid import uuid4

from raspbot_autonomy import AutonomousControl, RUNNING
from raspbot_context import MAX_TICK_GAP, context_rows, live_frames
from raspbot_recording import MODE_NAMES, parse_recording_packet

PORT = 5005
DEFAULT_INNER_RATIO = 0.5
TIMEOUT = 0.3


def motor_controller():
    vendor_path = Path('/home/pi/project_demo/lib')
    if vendor_path.is_dir():
        sys.path.append(str(vendor_path))
    # Use one low-level controller for every motion and both camera servos.
    module = importlib.import_module('Raspbot_Lib')
    return module.Raspbot()


ACTION_NAMES = {
    b'W': 'FORWARD', b'D': 'RIGHT', b'A': 'LEFT', b'STOP': 'STOP',
    b'WA': 'FORWARD_LEFT', b'WD': 'FORWARD_RIGHT',
    b'S': 'REVERSE', b'SA': 'REVERSE_LEFT', b'SD': 'REVERSE_RIGHT',
}
ACTIONS = tuple(ACTION_NAMES)
ARCS = (b'WA', b'WD', b'SA', b'SD')
COMMANDS = set(ACTIONS)
STRAIGHT_AND_SPIN_SIGNS = {
    b'W': (1, 1, 1, 1), b'S': (-1, -1, -1, -1),
    b'A': (-1, -1, 1, 1), b'D': (1, 1, -1, -1),
}


def validate_motor_ids(motor_ids):
    if len(motor_ids) != 4 or set(motor_ids) != {0, 1, 2, 3}:
        raise ValueError('motor IDs must be a permutation of 0, 1, 2, 3')


def arc_powers(command, speed, inner_ratio=DEFAULT_INNER_RATIO):
    """Return signed L1,L2,R1,R2 powers, not measured wheel velocities.

    Reverse steering keeps the original car-style semantics: SA turns the
    nose clockwise while reversing (the rear travels toward the left).
    """
    if command not in ARCS:
        raise ValueError('arc command must be WA, WD, SA or SD')
    if not isinstance(speed, int) or not 0 <= speed <= 255:
        raise ValueError('speed must be an integer from 0 to 255')
    if not 0 <= inner_ratio < 1:
        raise ValueError('inner ratio must be in [0, 1)')
    if speed == 0:
        return (0, 0, 0, 0)
    # Maintain the requested average motor command until a wheel reaches 255.
    outer = min(255, 2 * speed / (1 + inner_ratio))
    outer_power = round(outer)
    inner_power = round(outer * inner_ratio)
    left, right = (inner_power, outer_power) if command.endswith(b'A') else (outer_power, inner_power)
    if command.startswith(b'S'):
        left, right = -right, -left
    return (left, left, right, right)


def wheel_powers(command, speed, inner_ratio=DEFAULT_INNER_RATIO):
    if command in ARCS:
        return arc_powers(command, speed, inner_ratio)
    if command == b'STOP':
        return (0, 0, 0, 0)
    if not isinstance(speed, int) or not 0 <= speed <= 255:
        raise ValueError('speed must be an integer from 0 to 255')
    if command not in STRAIGHT_AND_SPIN_SIGNS:
        raise ValueError('unknown movement command')
    return tuple(sign * speed for sign in STRAIGHT_AND_SPIN_SIGNS[command])


def set_wheel_powers(controller, motor_ids, powers):
    validate_motor_ids(motor_ids)
    if len(powers) != 4 or any(
        not isinstance(power, int) or not -255 <= power <= 255
        for power in powers
    ):
        raise ValueError('four integer powers in [-255, 255] are required')
    for motor_id, power in zip(motor_ids, powers):
        controller.Ctrl_Muto(motor_id, power)


def drive_arc(controller, command, speed, motor_ids,
              inner_ratio=DEFAULT_INNER_RATIO):
    # Preserve this callable for existing imports.
    powers = arc_powers(command, speed, inner_ratio)
    set_wheel_powers(controller, motor_ids, powers)
    return powers


SAMPLE_PERIOD = 0.1


class CameraStream:
    def __init__(self, index):
        import cv2
        self.cv2 = cv2
        self.index = index
        self.frames = deque(maxlen=10)
        self.lock = Lock()
        self.closing = Event()
        self.ready = Event()
        self.error = None
        self.thread = Thread(target=self.capture, daemon=True)

    def capture(self):
        cap = None
        try:
            cap = self.cv2.VideoCapture(self.index)
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open camera {self.index}')
            cap.set(self.cv2.CAP_PROP_FPS, 30)
            while not self.closing.is_set():
                ok, frame = cap.read()
                timestamp = time.monotonic()
                if not ok:
                    raise RuntimeError('Camera stopped returning frames')
                image = self.cv2.resize(frame, (32, 32), interpolation=self.cv2.INTER_AREA)
                rgb = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.frames.append((timestamp, rgb.transpose(2, 0, 1)))
                    if len(self.frames) == 5:
                        self.ready.set()
        except Exception as error:
            self.error = error
            self.ready.set()
        finally:
            if cap is not None:
                cap.release()

    def snapshot(self, before=None):
        with self.lock:
            frames = tuple(self.frames)
        if before is not None:
            frames = tuple(frame for frame in frames if frame[0] <= before)
        return frames[-5:]

    def close(self):
        self.closing.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=1)


class Dataset:
    def __init__(self, path, limit, speed, pan, tilt, paired=False):
        import numpy as np
        self.np = np
        self.path = path
        self.paired = paired
        self.spool = tempfile.TemporaryDirectory(prefix='.recording-', dir=path) if paired else None
        shape = (limit, 5, 3, 32, 32)
        if paired:
            self.states = np.memmap(Path(self.spool.name) / 'frames.bin', mode='w+', shape=shape, dtype=np.uint8)
            self.modes = np.empty(limit, dtype=np.uint8)
            self.segment_ids = np.empty(limit, dtype=np.int32)
            self.context_segment_ids = np.empty(limit, dtype=np.int32)
        else:
            self.states = np.empty(shape, dtype=np.uint8)
        self.actions = np.empty(limit, dtype=np.uint8)
        self.timestamps = np.empty(limit, dtype=np.float64)
        self.frame_timestamps = np.empty((limit, 5), dtype=np.float64)
        self.metadata = dict(speed=speed, pan=pan, tilt=tilt, target_hz=10,
                             action_names=list(ACTION_NAMES.values()))
        self.count = 0
        self.started = None
        self.last_frame = None
        self.finish_frames = ()
        self.finish_time = None

    def mark_finish(self, frames, timestamp):
        if (self.count == 0 or len(frames) != 5
                or not 0 <= timestamp - frames[-1][0] < TIMEOUT):
            print('Finish not marked: no valid recent camera frames or driving samples.', flush=True)
            return
        self.finish_frames = frames
        self.finish_time = timestamp - self.started
        print('Finish marked from five camera frames.', flush=True)

    def append(self, frames, command, timestamp, mode=None, segment=None, context_segment=None):
        if command not in ACTIONS or frames[-1][0] == self.last_frame:
            return
        n = self.count
        for index, (captured, frame) in enumerate(frames):
            self.states[n, index] = frame
            self.frame_timestamps[n, index] = captured - self.started
        self.actions[n] = ACTIONS.index(command)
        if self.paired:
            self.modes[n] = mode
            self.segment_ids[n] = segment
            self.context_segment_ids[n] = segment if context_segment is None else context_segment
        self.timestamps[n] = timestamp - self.started
        self.last_frame = frames[-1][0]
        self.count += 1
        if self.count == 1 or self.count % 100 == 0:
            print(f'Samples: {self.count}/{len(self.actions)}', flush=True)

    def save(self):
        n = self.count
        if n == 0:
            print('No samples collected.', flush=True)
            return
        finish_states = self.np.empty((0, 5, 3, 32, 32), dtype=self.np.uint8)
        finish_frame_timestamps = self.np.empty((0, 5), dtype=self.np.float64)
        finish_timestamps = self.np.empty(0, dtype=self.np.float64)
        if self.finish_frames:
            finish_states = self.np.stack([image for _, image in self.finish_frames])[None]
            finish_frame_timestamps = self.np.array(
                [[captured - self.started for captured, _ in self.finish_frames]]
            )
            finish_timestamps = self.np.array([self.finish_time])
        payload = dict(
            states=self.states[:n], actions=self.actions[:n],
            timestamps=self.timestamps[:n], frame_timestamps=self.frame_timestamps[:n],
            finish_states=finish_states, finish_timestamps=finish_timestamps,
            finish_frame_timestamps=finish_frame_timestamps, **self.metadata,
        )
        if self.paired:
            paths = self.save_pair(payload)
            coverage = context_rows(self.frame_timestamps[:n], self.context_segment_ids[:n])
            common = (coverage['long'][:, 0] >= 0) & (coverage['hybrid'][:, 0] >= 0)
            print(f'Context coverage (all three layouts): {common.sum()}/{n} recorded ticks', flush=True)
            for path in paths:
                print(f'Saved {n} samples to {path.resolve()}', flush=True)
        else:
            self.write_outputs([(self.path, payload)])
            print(f'Saved {n} samples to {self.path.resolve()}', flush=True)
        counts = self.np.bincount(self.actions[:n], minlength=len(ACTIONS))
        print(dict(zip(ACTION_NAMES.values(), counts.tolist())), flush=True)

    def save_pair(self, payload):
        import fcntl

        with (self.path / '.numbering.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            legacy = next_dataset_path(self.path / 'legacy')
            labeled = next_dataset_path(self.path / 'mode')
            mode_payload = dict(
                payload, schema_version=2, session_id=uuid4().hex, legacy_filename=legacy.name,
                modes=self.modes[:self.count], mode_names=MODE_NAMES,
                segment_ids=self.segment_ids[:self.count],
                context_segment_ids=self.context_segment_ids[:self.count],
            )
            self.write_outputs([(legacy, payload), (labeled, mode_payload)])
        return legacy, labeled

    def write_outputs(self, outputs):
        staged = []
        published = []
        try:
            for path, payload in outputs:
                with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.partial', delete=False) as file:
                    staged.append((Path(file.name), path))
                    self.np.savez_compressed(file, **payload)
                    file.flush()
                    os.fsync(file.fileno())
            for temporary, path in staged:
                os.link(temporary, path)
                published.append(path)
        except BaseException:
            for path in published:
                path.unlink()
            raise
        finally:
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)

    def close(self):
        if self.spool is not None:
            del self.states
            self.spool.cleanup()


def next_dataset_path(directory):
    directory.mkdir(parents=True, exist_ok=True)
    numbers = []
    for path in directory.iterdir():
        match = re.fullmatch(r'(?:.*[_-])?(\d+)\.npz', path.name)
        if match:
            numbers.append(int(match[1]))
    return directory / f'run_{max(numbers, default=0) + 1:06d}.npz'


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=int, default=40)
    parser.add_argument('--pan', type=int, default=15)
    parser.add_argument('--tilt', type=int, default=100)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--record', type=Path, metavar='DIR_OR_NPZ',
                        help='Save paired datasets under a directory, or use .npz for legacy recording')
    parser.add_argument('--finish-on-interrupt', action='store_true',
                        help='Mark the finish point when the operator interrupts the receiver')
    parser.add_argument('--inner-ratio', type=float, default=DEFAULT_INNER_RATIO)
    parser.add_argument('--start-condition', default='unknown')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--finish-model', type=Path)
    parser.add_argument('--finish-threshold', type=float, default=0.8)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--samples', type=int, help='Maximum saved samples (defaults to recording minutes × 600)')
    parser.add_argument('--minutes', type=float, default=40)
    parser.add_argument('--motor-ids', type=int, nargs=4, default=(0, 1, 2, 3),
                        metavar=('LEFT1', 'LEFT2', 'RIGHT1', 'RIGHT2'))
    args = parser.parse_args()
    if not 0 <= args.inner_ratio < 1 or not 0 < args.finish_threshold < 1:
        parser.error('Invalid inner ratio or finish threshold')
    if args.model and args.record:
        parser.error('--model and --record are separate modes')
    if args.finish_model and not args.model:
        parser.error('--finish-model requires --model')
    if args.finish_on_interrupt and args.record is None:
        parser.error('--finish-on-interrupt requires --record')
    try:
        validate_motor_ids(args.motor_ids)
    except ValueError as error:
        parser.error(str(error))
    paired = args.record is not None and args.record.suffix.lower() != '.npz'
    if args.record is not None:
        if not 0 < args.minutes <= 40:
            parser.error('--minutes must be in (0, 40]')
        if args.samples is None:
            args.samples = max(1, round(args.minutes * 60 / SAMPLE_PERIOD)) if paired else 5000
        if args.samples < 1:
            parser.error('--samples must be positive')
        if paired:
            args.record.mkdir(parents=True, exist_ok=True)
        elif args.record.exists() or not args.record.parent.is_dir():
            parser.error('Legacy recording requires a new .npz file in an existing directory')
    if not 1 <= args.speed <= 255:
        parser.error('--speed must be between 1 and 255')
    if not 0 <= args.pan <= 180 or not 0 <= args.tilt <= 110:
        parser.error('--pan must be 0–180 and --tilt must be 0–110')

    signal.signal(signal.SIGTERM, terminate)
    policy = finish_policy = None
    if args.model:
        from raspbot_policy import Policy, state_input
        policy = Policy(args.model)
        if args.finish_model:
            finish_policy = Policy(args.finish_model, finish=True)
    bot = None if args.dry_run else motor_controller()
    dataset = Dataset(args.record, args.samples, args.speed, args.pan, args.tilt, paired=paired) if args.record else None
    if dataset is not None:
        dataset.metadata.update(start_condition=args.start_condition, inner_ratio=args.inner_ratio,
                                motor_ids=args.motor_ids, dry_run=args.dry_run)
    stream = CameraStream(args.camera) if dataset is not None or policy is not None else None

    autonomy = AutonomousControl(TIMEOUT) if policy is not None else None
    interrupted_at = None
    recording_mode = None
    controlled_recording = paired
    segment_started = float('inf')

    def drive(command):
        if bot is not None:
            if command in ARCS:
                drive_arc(bot, command, args.speed, args.motor_ids, args.inner_ratio)
            else:
                set_wheel_powers(bot, args.motor_ids, wheel_powers(command, args.speed))
        print(command.decode(), flush=True)

    try:
        drive(b'STOP')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as server:
            server.bind(('0.0.0.0', PORT))
            server.settimeout(0.05)
            if bot is not None:
                bot.Ctrl_Servo(1, args.pan)
                bot.Ctrl_Servo(2, args.tilt)
                print(f'Camera set to pan={args.pan}, tilt={args.tilt}', flush=True)
            if stream is not None:
                stream.thread.start()
                if not stream.ready.wait(timeout=5):
                    raise RuntimeError('Camera did not return five frames within five seconds')
                if stream.error is not None:
                    raise RuntimeError('Camera initialization failed') from stream.error
                if policy is not None:
                    print('ONNX mode: use keyboard --auto. Space or the button starts and stops driving.', flush=True)
                elif paired:
                    print('Recording OFF. Use keyboard --record, then R to start or pause, M to change mode. Ctrl+C saves.', flush=True)
                else:
                    print('Single NPZ: plain keyboard starts recording on movement. With keyboard --record, use R to start or pause.', flush=True)

            print(f'Listening on UDP {PORT}', flush=True)
            active = desired = b'STOP'
            last_command = float('-inf')
            next_tick = time.monotonic()
            finish_hits = 0
            finished = False
            last_policy_frame = None
            policy_history = deque(maxlen=24)
            last_predicted_mode = None
            reverse_since = None
            recording_source = None
            recording_revision = -1
            segment_id = 0
            context_segment_id = 0
            if stream is not None:
                server.settimeout(0.01)

            while True:
                sender = None
                try:
                    packet, sender = server.recvfrom(128)
                    command = packet
                except socket.timeout:
                    packet = command = None

                now = time.monotonic()
                if now - last_command >= TIMEOUT:
                    desired = b'STOP'
                    if autonomy is not None:
                        autonomy.expire(now)
                    recording_source = None
                    recording_mode = None
                    if active != b'STOP':
                        drive(b'STOP')
                        active = b'STOP'
                        print('Command timeout', flush=True)

                tagged = parse_recording_packet(packet)
                if dataset is not None and tagged is not None:
                    candidate, revision, mode = tagged
                    if candidate in COMMANDS and (
                        recording_source is None or
                        (sender == recording_source and revision >= recording_revision)
                    ):
                        if recording_source is None or revision != recording_revision or mode != recording_mode:
                            if mode is not None:
                                continuous_mode_switch = (sender == recording_source and recording_mode is not None
                                                          and mode != recording_mode
                                                          and revision == recording_revision + 1)
                                if not continuous_mode_switch:
                                    context_segment_id += 1
                                segment_id += 1
                                segment_started = now
                            print(f'Recording: {MODE_NAMES[mode] if mode is not None else "OFF"}', flush=True)
                        controlled_recording = True
                        recording_source = sender
                        recording_revision = revision
                        recording_mode = mode
                        command = candidate
                elif controlled_recording and command in COMMANDS:
                    # Plain packets may stop a controlled recording run but never arm recording.
                    if command == b'STOP':
                        recording_source = None
                        recording_mode = None
                    else:
                        command = None

                reply = None
                if autonomy is not None:
                    reply = autonomy.receive(packet, sender, now, finished=finished)
                    if reply is not None and reply.startswith(RUNNING):
                        last_command = now
                    if not autonomy.started:
                        policy_history.clear()
                        last_policy_frame = None
                        if active != b'STOP':
                            drive(b'STOP')
                            active = b'STOP'
                        finish_hits = 0
                        reverse_since = None
                elif command in COMMANDS:
                    last_command = now
                    desired = command
                    if (stream is None or desired == b'STOP') and desired != active:
                        drive(desired)
                        active = desired
                    reply = packet
                if reply is not None:
                    try:
                        server.sendto(reply, sender)
                    except OSError:
                        pass

                if stream is None:
                    continue
                if stream.error is not None:
                    raise RuntimeError('Camera capture failed') from stream.error
                if dataset is not None and dataset.started is not None and now - dataset.started >= args.minutes * 60:
                    break
                if now < next_tick:
                    continue
                next_tick += SAMPLE_PERIOD
                if next_tick <= now:
                    next_tick = now + SAMPLE_PERIOD

                frames = stream.snapshot()
                timestamp = time.monotonic()
                if len(frames) != 5 or timestamp - frames[-1][0] >= TIMEOUT:
                    raise RuntimeError('Camera frames are stale')
                if timestamp - last_command >= TIMEOUT:
                    if autonomy is not None:
                        autonomy.expire(timestamp)
                    if active != b'STOP':
                        drive(b'STOP')
                        active = b'STOP'
                    desired = b'STOP'
                    continue

                if policy is not None:
                    if not autonomy.started or finished:
                        continue
                    if frames[-1][0] == last_policy_frame:
                        continue
                    last_policy_frame = frames[-1][0]
                    if policy_history and last_policy_frame - policy_history[-1][0] > MAX_TICK_GAP:
                        policy_history.clear()
                    policy_history.append(frames[-1])
                    selected = live_frames(policy_history, frames, policy.context)
                    if selected is None:
                        if active != b'STOP':
                            drive(b'STOP')
                            active = b'STOP'
                        reverse_since = None
                        finish_hits = 0
                        continue
                    target, predicted_mode = policy.predict(state_input(selected))
                    if predicted_mode != last_predicted_mode:
                        print(f'Predicted mode: {predicted_mode}', flush=True)
                        last_predicted_mode = predicted_mode
                    if finish_policy is not None:
                        finish_x = state_input([image for _, image in frames])
                        finish_hits = (finish_hits + 1
                                       if finish_policy.finish_score(finish_x) >= args.finish_threshold else 0)
                        if finish_hits >= 2:
                            finished = True
                            autonomy.stop()
                            target = b'STOP'
                            print('Finish detected. Restart receiver for the next run.', flush=True)
                    decision_time = time.monotonic()
                    if target in (b'S', b'SA', b'SD'):
                        if reverse_since is None:
                            reverse_since = decision_time
                        if decision_time - reverse_since >= 0.5:
                            drive(b'STOP')
                            print('Reverse limit reached. Run stopped.', flush=True)
                            break
                    else:
                        reverse_since = None
                    autonomy.expire(decision_time)
                    if decision_time - frames[-1][0] >= TIMEOUT:
                        autonomy.stop()
                    if not autonomy.started:
                        target = b'STOP'
                        finish_hits = 0
                        reverse_since = None
                    if target != active:
                        drive(target)
                        active = target
                    continue

                if desired != active:
                    drive(desired)
                    active = desired
                if controlled_recording:
                    if recording_mode is None or frames[0][0] < segment_started:
                        continue
                    if dataset.started is None and (paired or active != b'STOP'):
                        dataset.started = timestamp
                elif dataset.started is None and active != b'STOP':
                    dataset.started = timestamp
                if dataset.started is not None:
                    dataset.append(frames, active, timestamp, recording_mode, segment_id, context_segment_id)
                    if dataset.count == args.samples:
                        break
    except KeyboardInterrupt:
        interrupted_at = time.monotonic()
    finally:
        try:
            drive(b'STOP')
        finally:
            if stream is not None:
                try:
                    if (args.finish_on_interrupt and interrupted_at is not None
                            and stream.error is None):
                        frames = stream.snapshot(before=interrupted_at)
                        if not controlled_recording or (recording_mode is not None and frames and frames[0][0] >= segment_started):
                            dataset.mark_finish(frames, interrupted_at)
                        else:
                            print('Finish not marked: controlled recording was off or switching modes.', flush=True)
                finally:
                    stream.close()
                    if dataset is not None:
                        try:
                            dataset.save()
                        finally:
                            dataset.close()


if __name__ == '__main__':
    main()
