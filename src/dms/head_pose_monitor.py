import time
import numpy as np


class HeadPoseMonitor:
    def __init__(
        self,
        yaw_threshold=20.0,
        pitch_threshold=15.0,
        roll_threshold=15.0,
        sustained_duration=2.0,
    ):
        # Head-pose deviation thresholds in degrees
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold

        # Minimum continuous duration for sustained deviation
        self.sustained_duration = sustained_duration

        # Time when the head deviation started
        self.deviation_start_time = None

    def update(self, head_pose_deviation, tracking_valid=True):
        """
        Monitor head-pose deviation from the calibrated baseline.

        Args:
            head_pose_deviation: Absolute deviation [yaw, pitch, roll].
            tracking_valid: Whether the current tracking result is valid.

        Returns:
            Dictionary containing status and deviation duration.
        """

        # Ignore invalid tracking observations
        if not tracking_valid or head_pose_deviation is None:
            self.deviation_start_time = None

            return {
                "status": "UNKNOWN",
                "duration": 0.0,
            }

        # Validate the deviation values
        deviation = np.asarray(head_pose_deviation, dtype=float)

        if deviation.shape != (3,) or not np.all(np.isfinite(deviation)):
            self.deviation_start_time = None

            return {
                "status": "UNKNOWN",
                "duration": 0.0,
            }

        # Extract deviations for all three angles
        yaw_deviation = deviation[0]
        pitch_deviation = deviation[1]
        roll_deviation = deviation[2]

        # Check whether any angle exceeds its threshold
        head_turned = (
            yaw_deviation > self.yaw_threshold
            or pitch_deviation > self.pitch_threshold
            or roll_deviation > self.roll_threshold
        )

        current_time = time.monotonic()

        # Reset the timer when the head returns to the normal range
        if not head_turned:
            self.deviation_start_time = None

            return {
                "status": "SAFE",
                "duration": 0.0,
            }

        # Start timing when the deviation first appears
        if self.deviation_start_time is None:
            self.deviation_start_time = current_time

        # Calculate continuous deviation duration
        duration = current_time - self.deviation_start_time

        # Classify the observation
        if duration >= self.sustained_duration:
            status = "LOW_AWARENESS"
        else:
            status = "MONITORING"

        return {
            "status": status,
            "duration": duration,
        }