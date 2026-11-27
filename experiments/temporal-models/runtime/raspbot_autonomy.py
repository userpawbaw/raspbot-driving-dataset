from collections import deque
from uuid import uuid4


START = b'AUTO:START:'
KEEP = b'AUTO:KEEP:'
STOP = b'AUTO:STOP:'
RUNNING = b'AUTO:RUNNING:'
STOPPED = b'AUTO:STOPPED:'


def run_token():
    return uuid4().hex.encode('ascii')


class AutonomousControl:
    def __init__(self, timeout):
        self.timeout = timeout
        self.token = None
        self.source = None
        self.deadline = float('-inf')
        self.started = False
        self.retired = deque(maxlen=64)

    def stop(self):
        if self.token is not None:
            self.retired.append(self.token)
        self.token = self.source = None
        self.started = False

    def expire(self, now):
        if self.token is not None and now >= self.deadline:
            self.stop()

    def receive(self, packet, sender, now, *, finished=False):
        self.expire(now)
        if packet == b'STOP':
            self.stop()
            return packet

        operation, _, token = (packet or b'').rpartition(b':')
        if len(token) != 32 or any(byte not in b'0123456789abcdef' for byte in token):
            return None
        if operation == STOP[:-1]:
            if (sender, token) == (self.source, self.token):
                self.stop()
            elif token != self.token:
                self.retired.append(token)
            return STOPPED + token
        if operation == START[:-1]:
            if finished or self.token is not None or token in self.retired:
                return STOPPED + token
            self.token, self.source = token, sender
            self.deadline = now + self.timeout
            # Arm on the start packet. A heartbeat confirms the controller is still enabled.
            return None
        if operation == KEEP[:-1]:
            if (sender, token) == (self.source, self.token):
                self.started = True
                self.deadline = now + self.timeout
                return RUNNING + token
            return STOPPED + token
        return None
