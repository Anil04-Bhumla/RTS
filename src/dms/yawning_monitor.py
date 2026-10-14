import time


class YawningMonitor:
    def __init__(
        self,
        confirmation_duration=2.0,
    ):
        self.confirmation_duration = confirmation_duration
        self.open_start_time = None

    def update(self, mouth_state, tracking_valid=True):
        """
        Mouth classes:
        0 = Closed
        1 = Semi-open
        2 = Open
        """

        if (
            not tracking_valid
            or mouth_state not in (0, 1, 2)
        ):
            self.open_start_time = None

            return {
                "status": "UNKNOWN",
                "duration": 0.0,
            }

        current_time = time.monotonic()

        # Closed mouth resets the timer
        if mouth_state == 0:
            self.open_start_time = None

            return {
                "status": "SAFE",
                "duration": 0.0,
            }

        # Start timing when mouth becomes open or semi-open
        if self.open_start_time is None:
            self.open_start_time = current_time

        duration = current_time - self.open_start_time

        if duration >= self.confirmation_duration:
            status = "DETECTED"
        else:
            status = "MONITORING"

        return {
            "status": status,
            "duration": duration,
        }