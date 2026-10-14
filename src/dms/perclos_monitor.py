from collections import deque
import time

import numpy as np


class PerclosMonitor:
    """
    Rolling-window PERCLOS monitor.

    Provisional eye-output mapping:
        eyes[0], eyes[1] -> eye visibility
        eyes[2], eyes[3] -> eye opening

    This mapping and the thresholds must be validated experimentally.
    """

    def __init__(
        self,
        window_seconds=60.0,
        opening_threshold=0.20,
        visibility_threshold=0.50,
        max_sample_gap_seconds=0.25,
    ):
        self.window_seconds = window_seconds
        self.opening_threshold = opening_threshold
        self.visibility_threshold = visibility_threshold
        self.max_sample_gap_seconds = max_sample_gap_seconds

        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def update(self, eyes, tracking_valid=True):
        now = time.monotonic()

        # Missing face tracking or invalid output means UNKNOWN.
        if not tracking_valid:
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "reason": "Tracking invalid",
            }

        if eyes is None:
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "reason": "Eye output missing",
            }

        try:
            values = np.asarray(eyes, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            values = np.array([])

        if values.size != 4 or not np.all(np.isfinite(values)):
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "reason": "Invalid eye output",
            }

        visibility = values[:2]
        opening = values[2:]

        if (
            np.any((visibility < 0) | (visibility > 1))
            or np.any((opening < 0) | (opening > 1))
        ):
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "reason": "Eye values outside expected range",
            }

        # Don't infer eye closure when an eye is not sufficiently visible.
        if np.any(visibility < self.visibility_threshold):
            return {
                "eye_state": "UNKNOWN",
                "perclos": None,
                "reason": "Eye visibility insufficient",
            }

        # Both eyes must be below the provisional threshold.
        closed = bool(np.all(opening <= self.opening_threshold))

        self.samples.append((now, closed))

        cutoff = now - self.window_seconds

        # Retain one sample before the cutoff to clip the window correctly.
        while len(self.samples) >= 2 and self.samples[1][0] <= cutoff:
            self.samples.popleft()

        closed_seconds = 0.0
        observed_seconds = 0.0

        records = list(self.samples)

        for i in range(len(records) - 1):
            t0, closed0 = records[i]
            t1, _ = records[i + 1]

            gap = t1 - t0

            # Do not count time across long gaps.
            if gap <= 0 or gap > self.max_sample_gap_seconds:
                continue

            duration = min(t1, now) - max(t0, cutoff)

            if duration <= 0:
                continue

            observed_seconds += duration

            if closed0:
                closed_seconds += duration

        perclos = None

        if observed_seconds > 0:
            perclos = 100.0 * closed_seconds / observed_seconds

        return {
            "eye_state": "CLOSED" if closed else "OPEN",
            "perclos": perclos,
            "reason": "Valid eye observation",
        }