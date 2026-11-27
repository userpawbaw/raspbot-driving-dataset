"""Causal frame selection shared by offline training and live driving."""
from bisect import bisect_left

SHORT_FRAMES = 5
FRAME_SHAPE = (3, 32, 32)
MAX_TICK_GAP = 0.25
MAX_OFFSET_ERROR = 0.085
HISTORY_OFFSETS = {
    'short': (),
    'long': (1.0, 0.75, 0.50, 0.25),
    'hybrid': (1.0, 0.8, 0.6, 0.4, 0.2),
}


def frame_count(layout):
    return len(HISTORY_OFFSETS[layout]) + (1 if layout == 'long' else SHORT_FRAMES)


def history_indices(times, end, start, layout):
    """Find past tick snapshots (in chronological order) relative to tick end."""
    offsets = HISTORY_OFFSETS[layout]
    if not offsets:
        return ()
    selected = []
    for offset in offsets:
        target = times[end] - offset
        right = bisect_left(times, target, start, end)
        candidates = range(max(start, right - 1), min(end, right + 1))
        nearest = min(candidates, key=lambda row: abs(times[row] - target), default=None)
        if (nearest is None or abs(times[nearest] - target) > MAX_OFFSET_ERROR
                or (selected and nearest <= selected[-1])):
            return None
        selected.append(nearest)
    return tuple(selected)


def context_rows(frame_times, segments):
    """Map each tick to past rows, or -1 if a continuous context is unavailable."""
    import numpy as np

    if (frame_times.shape != (len(segments), SHORT_FRAMES)
            or not np.isfinite(frame_times).all() or np.any(np.diff(frame_times, axis=1) <= 0)):
        raise ValueError('invalid frame_timestamps')
    latest = frame_times[:, -1]
    deltas = np.diff(latest)
    if np.any(deltas <= 0):
        raise ValueError('latest frame_timestamps must increase between ticks')

    rows_by_layout = {
        layout: np.full((len(segments), len(offsets)), -1, dtype=np.int64)
        for layout, offsets in HISTORY_OFFSETS.items() if offsets
    }
    start = 0
    for row in range(len(segments)):
        if row and (segments[row] != segments[row - 1] or deltas[row - 1] > MAX_TICK_GAP):
            start = row
        for layout, rows in rows_by_layout.items():
            indices = history_indices(latest, row, start, layout)
            if indices is not None and (layout != 'hybrid' or latest[indices[-1]] < frame_times[row, 0]):
                rows[row] = indices
    return rows_by_layout


def live_frames(history, recent, layout):
    if layout == 'short':
        return [frame for _, frame in recent]
    times = [time for time, _ in history]
    indices = history_indices(times, len(times) - 1, 0, layout)
    if indices is None or (layout == 'hybrid' and times[indices[-1]] >= recent[0][0]):
        return None
    current = recent[-1:] if layout == 'long' else recent
    return [history[index][1] for index in indices] + [frame for _, frame in current]
