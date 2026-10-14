import numpy as np


class DriverCalibration:
    """
    Establishes a driver's baseline head pose.

    Calibration should be performed while the driver is
    alert and looking in a normal driving posture.
    """

    def __init__(self, required_samples=60):
        if required_samples < 1:
            raise ValueError("required_samples must be >= 1")

        self.required_samples = required_samples
        self.head_pose_samples = []

        self.baseline_head_pose = None
        self.is_calibrated = False

    def update(self, euler, tracking_valid=True):
        """
        Collect valid head-pose samples until calibration completes.

        euler: [yaw, pitch, roll] in degrees.
        tracking_valid: True only when the face track and
                         head-pose prediction are valid.
        """

        if self.is_calibrated:
            return {
                "status": "CALIBRATED",
                "progress": 100.0,
                "baseline_head_pose": self.baseline_head_pose.copy(),
            }

        if not tracking_valid:
            return self._progress_result()

        pose = np.asarray(euler, dtype=float)

        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return self._progress_result()

        self.head_pose_samples.append(pose.copy())

        if len(self.head_pose_samples) >= self.required_samples:
            samples = np.asarray(self.head_pose_samples)

            self.baseline_head_pose = np.median(samples, axis=0)
            self.is_calibrated = True

            return {
                "status": "CALIBRATED",
                "progress": 100.0,
                "baseline_head_pose": self.baseline_head_pose.copy(),
            }

        return self._progress_result()

    def _progress_result(self):
        progress = (
            len(self.head_pose_samples) / self.required_samples
        ) * 100.0

        return {
            "status": "CALIBRATING",
            "progress": min(progress, 100.0),
            "baseline_head_pose": None,
        }

    def get_head_pose_deviation(self, euler):
        """
        Return absolute yaw, pitch and roll deviations
        from the calibrated baseline.
        """

        if not self.is_calibrated:
            return None

        pose = np.asarray(euler, dtype=float)

        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return None

        return np.abs(pose - self.baseline_head_pose)