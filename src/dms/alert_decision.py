class AlertDecision:
    """
    PERCLOS-based driver alert decision module.
    Thresholds are initial experimental values.
    """

    def __init__(self, warning_threshold=15.0, danger_threshold=25.0):
        if not 0 <= warning_threshold < danger_threshold <= 100:
            raise ValueError(
                "Thresholds must satisfy "
                "0 <= warning < danger <= 100"
            )

        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold

    def update(self, perclos, eye_state):
        if perclos is None or eye_state == "UNKNOWN":
            return {
                "status": "UNKNOWN",
                "message": "Insufficient valid observations",
            }

        if perclos >= self.danger_threshold:
            return {
                "status": "DANGER",
                "message": "High eye-closure level detected",
            }

        if perclos >= self.warning_threshold:
            return {
                "status": "WARNING",
                "message": "Elevated eye-closure level",
            }

        return {
            "status": "SAFE",
            "message": "No PERCLOS threshold exceeded",
        }