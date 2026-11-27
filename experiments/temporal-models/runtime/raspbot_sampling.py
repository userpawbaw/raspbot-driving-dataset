"""Timestamp-based sampling for the eight Raspbot input layouts."""
from bisect import bisect_left
from collections import deque

LAYOUTS = {
    'recent5': (5, 0, 0.0), 'recent10': (10, 0, 0.0), 'recent15': (15, 0, 0.0),
    'stride015': (1, 4, 0.15), 'stride020': (1, 4, 0.2), 'stride030': (1, 4, 0.3),
    'hybrid030': (5, 5, 0.3), 'hybrid020': (5, 5, 0.2),
}
MAX_FRAME_GAP = 0.12
MAX_TICK_GAP = 0.25
TOLERANCE = 0.055


def indices(times, end, start, layout):
    recent, older, step = LAYOUTS[layout]
    first = end - recent + 1
    if first < start:
        return None
    chosen = []
    upper = first
    for offset in range(1, older + 1):
        target = times[first] - offset * step
        pos = bisect_left(times, target, start, upper)
        candidates = range(max(start, pos - 1), min(upper, pos + 1))
        best = min(candidates, key=lambda i: abs(times[i] - target), default=None)
        if best is None or abs(times[best] - target) > TOLERANCE:
            return None
        chosen.append(best)
        upper = best
    return tuple(reversed(chosen)) + tuple(range(first, end + 1))


class LiveSampler:
    def __init__(self):
        self.frames = deque(maxlen=256)
        self.latest_tick = None

    def clear(self):
        self.frames.clear()
        self.latest_tick = None

    def select(self, recent, layout):
        if self.latest_tick is not None and recent[-1][0] - self.latest_tick > MAX_TICK_GAP:
            self.clear()
            # Discard frames already present before the missing tick interval.
            recent = recent[-1:]
        self.latest_tick = recent[-1][0]
        for item in recent:
            if self.frames and item[0] <= self.frames[-1][0]:
                continue
            if self.frames and item[0] - self.frames[-1][0] > MAX_FRAME_GAP:
                self.frames.clear()
            self.frames.append(item)
        times = [item[0] for item in self.frames]
        selected = indices(times, len(times) - 1, 0, layout)
        return None if selected is None else [self.frames[i][1] for i in selected]
