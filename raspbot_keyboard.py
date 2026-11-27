import argparse
import socket
import time
import tkinter as tk

from raspbot_autonomy import KEEP, RUNNING, START, STOP, STOPPED, run_token
from raspbot_recording import MODE_NAMES, recording_packet

PORT = 5005
KEYS = {'w': 'w', 's': 's', 'a': 'a', 'd': 'd',
        'up': 'w', 'down': 's', 'left': 'a', 'right': 'd'}


def command_for(pressed):
    if not pressed:
        return b'STOP'
    return KEYS[next(reversed(pressed))].upper().encode('ascii')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('robot_ip')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--record', action='store_true', help='Enable recording controls')
    modes.add_argument('--auto', action='store_true', help='Toggle ONNX driving with Space or the button')
    args = parser.parse_args()

    window = tk.Tk()
    window.title('Raspbot keyboard test')
    x11 = window.tk.call('tk', 'windowingsystem') == 'x11'
    instructions = ('Space starts or stops ONNX driving.' if args.auto else
                    'Use WASD or arrow keys. Last pressed direction wins. Space stops.')
    tk.Label(window, text=instructions, padx=20, pady=20).pack()
    intent = tk.StringVar(value='ONNX driving stopped' if args.auto else 'Sent: STOP')
    tk.Label(window, textvariable=intent).pack()
    status = tk.StringVar(value='Waiting for robot reply')
    tk.Label(window, textvariable=status, padx=20, pady=15).pack()

    recording_status = tk.StringVar()
    if args.record:
        tk.Label(window, text='R: recording on/off    M: normal/recovery').pack()
        tk.Label(window, textvariable=recording_status, padx=20, pady=10).pack()

    enabled = False
    mode = 0
    revision = 0
    controls_held = set()
    control_releases = {}
    pressed = {}  # Insertion order follows physical key presses.
    movement_releases = {}
    blocked = set()
    latest_ack = b''
    last_reply = float('-inf')
    auto_token = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.connect((args.robot_ip, PORT))
        client.setblocking(False)

        def set_recording(new_enabled, new_mode):
            nonlocal enabled, mode, revision
            if (enabled, mode) != (new_enabled, new_mode):
                revision += 1
                enabled, mode = new_enabled, new_mode

        def send(command):
            intent.set('Sent: ' + command.decode())
            try:
                packet = recording_packet(command, revision, enabled, mode) if args.record else command
                client.send(packet)
            except OSError:
                pass

        def send_auto(packet):
            try:
                client.send(packet)
            except OSError:
                pass

        def command():
            return command_for(pressed)

        def stop(event=None):
            nonlocal auto_token
            if args.auto:
                send_auto(STOP + auto_token if auto_token is not None else b'STOP')
                auto_token = None
                auto_button.configure(text='Start')
                intent.set('ONNX driving stopped')
                return
            set_recording(False, mode)
            blocked.update(pressed)
            pressed.clear()
            send(b'STOP')

        def toggle_auto():
            nonlocal auto_token
            if auto_token is not None:
                stop()
            elif time.monotonic() - last_reply <= 0.5:
                auto_token = run_token()
                auto_button.configure(text='Stop')
                intent.set('Waiting for ONNX start confirmation')
                send_auto(START + auto_token)

        if args.auto:
            auto_button = tk.Button(window, text='Start', command=toggle_auto, takefocus=False)
            auto_button.pack(pady=(0, 15))

        def on_press(event):
            key = event.keysym.lower()
            if args.auto and key == 'space':
                if key in control_releases:
                    window.after_cancel(control_releases.pop(key))
                if key not in controls_held:
                    controls_held.add(key)
                    toggle_auto()
                return 'break'
            if args.auto:
                return
            if key == 'space':
                stop()
            elif args.record and key in ('r', 'm'):
                if key in control_releases:
                    window.after_cancel(control_releases.pop(key))
                if key not in controls_held:
                    controls_held.add(key)
                    if key == 'r' and time.monotonic() - last_reply <= 0.5:
                        set_recording(not enabled, mode)
                    elif key == 'm':
                        set_recording(enabled, 1 - mode)
                    send(command())
            elif key in KEYS:
                pending = movement_releases.pop(key, None)
                if pending is not None:
                    timer, released_at = pending
                    window.after_cancel(timer)
                    if released_at == event.time:
                        return
                    blocked.discard(key)
                    pressed.pop(key, None)
                if key not in pressed and key not in blocked:
                    pressed[key] = None
                    send(command())

        def release_control(key):
            controls_held.discard(key)
            control_releases.pop(key, None)

        def on_release(event):
            key = event.keysym.lower()
            if key in controls_held and key not in control_releases:
                # X11 can emit a release/press pair on key repeat.
                control_releases[key] = window.after(60, release_control, key)
            if key not in pressed and key not in blocked:
                return
            if x11:
                # X11 repeat can synthesize a release and press at the same timestamp.
                if key not in movement_releases:
                    timer = window.after(10, release_movement, key)
                    movement_releases[key] = (timer, event.time)
            else:
                release_movement(key)

        def release_movement(key):
            movement_releases.pop(key, None)
            blocked.discard(key)
            if key in pressed:
                del pressed[key]
                send(command())

        def update():
            nonlocal latest_ack, last_reply
            for _ in range(32):
                try:
                    latest_ack = client.recv(128)
                    last_reply = time.monotonic()
                    if auto_token is not None:
                        if latest_ack == STOPPED + auto_token:
                            stop()
                        elif latest_ack == RUNNING + auto_token:
                            intent.set('ONNX driving active')
                except OSError:
                    break
            if time.monotonic() - last_reply > 0.5:
                stop()
                status.set('No reply from Raspbot')
            elif args.auto:
                status.set('Connected to Raspbot')
            else:
                status.set('Raspbot received: ' + latest_ack.decode('ascii', errors='replace'))
            if args.record:
                expected = recording_packet(command(), revision, enabled, mode).rsplit(b':', 1)[0]
                confirmed = time.monotonic() - last_reply <= 0.5 and latest_ack.startswith(expected + b':')
                label = MODE_NAMES[mode]
                state = 'ON' if enabled else 'OFF'
                recording_status.set(f'Recording {state} | {label} | ' + ('confirmed' if confirmed else 'waiting for receiver'))
            if args.auto:
                send_auto(KEEP + auto_token if auto_token is not None else b'STOP')
            else:
                send(command())
            window.after(100, update)

        window.bind('<KeyPress>', on_press)
        window.bind('<KeyRelease>', on_release)
        window.bind('<FocusOut>', stop)
        def on_focus(event):
            if args.auto:
                return
            blocked.clear()
            for timer, _ in movement_releases.values():
                window.after_cancel(timer)
            movement_releases.clear()
            controls_held.clear()
            for pending in control_releases.values():
                window.after_cancel(pending)
            control_releases.clear()

        window.bind('<FocusIn>', on_focus)
        window.bind('<Escape>', lambda event: window.destroy())
        window.focus_set()
        try:
            update()
            window.mainloop()
        finally:
            try:
                if args.auto and auto_token is not None:
                    packet = STOP + auto_token
                else:
                    packet = recording_packet(b'STOP', revision + 1, False, mode) if args.record else b'STOP'
                client.send(packet)
            except OSError:
                pass


if __name__ == '__main__':
    main()
