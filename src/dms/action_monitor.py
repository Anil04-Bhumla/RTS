import time


class ActionMonitor:
    def __init__(
        self,
        confirmation_duration=1.0,
        history_duration=2.0,
        minimum_detection_ratio=0.7,
    ):
        self.confirmation_duration = confirmation_duration
        self.history_duration = history_duration
        self.minimum_detection_ratio = minimum_detection_ratio

        self.history = []
        self.last_timestamp = None

    def update(self, action, tracking_valid=True):
        current_time = time.monotonic()

        if not tracking_valid or action not in (0, 1, 2):
            self.history.clear()
            self.last_timestamp = None

            return {
                "status": "UNKNOWN",
                "action": None,
                "duration": 0.0,
            }

        # Remove observations if time moved backwards
        if (
            self.last_timestamp is not None
            and current_time < self.last_timestamp
        ):
            self.history.clear()

        self.last_timestamp = current_time

        # Store the current valid prediction
        self.history.append((current_time, action))

        # Keep only observations from the recent history window
        cutoff = current_time - self.history_duration
        self.history = [
            item for item in self.history
            if item[0] >= cutoff
        ]

        # Identify the most recent non-normal action
        non_normal = [
            item for item in self.history
            if item[1] in (1, 2)
        ]

        if not non_normal:
            return {
                "status": "SAFE",
                "action": "Normal",
                "duration": 0.0,
            }

        # Use the latest non-normal class as the candidate action
        candidate = non_normal[-1][1]

        # Count matching predictions in the recent history
        matching = [
            item for item in self.history
            if item[1] == candidate
        ]

        total = len(self.history)
        ratio = len(matching) / total if total else 0.0

        # Measure how long the candidate has been observed
        candidate_start = matching[0][0]
        duration = current_time - candidate_start

        confirmed = (
            duration >= self.confirmation_duration
            and ratio >= self.minimum_detection_ratio
        )

        if confirmed:
            status = "DETECTED"
        else:
            status = "MONITORING"

        action_name = {
            0: "Normal",
            1: "Phone Use",
            2: "Smoking",
        }[candidate]

        return {
            "status": status,
            "action": action_name,
            "duration": duration,
        }