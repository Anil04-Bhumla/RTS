class DecisionUnit:
    """
    Combines PERCLOS, head-pose, action and mouth-monitor
    outputs into a single driver-risk decision.

    Thresholds are initial experimental values.
    """

    RISK_LEVELS = {
        "SAFE": 0,
        "MONITORING": 1,
        "WARNING": 1,
        "LOW_AWARENESS": 1,
        "DETECTED": 2,
        "DANGER": 2,
    }

    def __init__(self, cooling_period=2.0):
        self.cooling_period = cooling_period
        self.current_status = "CALIBRATION"
        self.lower_risk_since = None

    def update(
        self,
        perclos_result,
        head_pose_result,
        action_result,
        mouth_result,
        tracking_valid=True,
        calibrated=True,
    ):
        import time

        # Calibration has priority
        if not calibrated:
            self.current_status = "CALIBRATION"
            self.lower_risk_since = None

            return {
                "status": "CALIBRATION",
                "message": "Calibrating driver baseline",
            }

        # Do not make a risk decision from invalid tracking
        if not tracking_valid:
            self.current_status = "UNKNOWN"
            self.lower_risk_since = None

            return {
                "status": "UNKNOWN",
                "message": "Insufficient valid tracking",
            }

        # Extract individual monitor statuses
        perclos_status = perclos_result.get("status", "UNKNOWN")
        head_status = head_pose_result.get("status", "UNKNOWN")
        action_status = action_result.get("status", "UNKNOWN")
        mouth_status = mouth_result.get("status", "UNKNOWN")

        # Convert PERCLOS status to the shared risk vocabulary
        if perclos_status == "WARNING":
            perclos_risk = "WARNING"
        elif perclos_status == "DANGER":
            perclos_risk = "DANGER"
        elif perclos_status == "SAFE":
            perclos_risk = "SAFE"
        else:
            perclos_risk = "UNKNOWN"

        # Map individual signals to risk levels
        risks = {
            "perclos": perclos_risk,
            "head_pose": head_status,
            "action": action_status,
            "mouth": mouth_status,
        }

        # UNKNOWN observations are not treated as SAFE.
        # If all modules are unknown, the overall state is unknown.
        known_risks = [
            status for status in risks.values()
            if status != "UNKNOWN"
        ]

        if not known_risks:
            self.current_status = "UNKNOWN"
            self.lower_risk_since = None

            return {
                "status": "UNKNOWN",
                "message": "No valid risk observations",
            }

        highest_risk = max(
            known_risks,
            key=lambda status: self.RISK_LEVELS.get(status, 0),
        )

        # Map the highest risk to the global driver status
        if self.RISK_LEVELS.get(highest_risk, 0) >= 2:
            target_status = "DANGER"
        elif self.RISK_LEVELS.get(highest_risk, 0) == 1:
            target_status = "LOW_AWARENESS"
        else:
            target_status = "SAFE"

        current_time = time.monotonic()

        # Increase risk immediately; delay transitions to lower risk
        current_level = self.RISK_LEVELS.get(self.current_status, -1)
        target_level = self.RISK_LEVELS.get(target_status, 0)

        if target_level > current_level:
            self.current_status = target_status
            self.lower_risk_since = None

        elif target_level < current_level:
            if self.lower_risk_since is None:
                self.lower_risk_since = current_time

            if current_time - self.lower_risk_since >= self.cooling_period:
                self.current_status = target_status
                self.lower_risk_since = None

        else:
            self.lower_risk_since = None
            self.current_status = target_status

        return {
            "status": self.current_status,
            "message": f"Combined risk from monitors: {highest_risk}",
            "signals": risks,
        }