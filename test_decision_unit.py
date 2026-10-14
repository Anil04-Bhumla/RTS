from src.dms.decision_unit import DecisionUnit

decision_unit = DecisionUnit(cooling_period=0.1)

# Test 1: Driver calibration is incomplete
result = decision_unit.update(
    perclos_result={"status": "SAFE"},
    head_pose_result={"status": "SAFE"},
    action_result={"status": "SAFE"},
    mouth_result={"status": "SAFE"},
    tracking_valid=True,
    calibrated=False,
)
print("Test 1 - Calibration:", result["status"])

# Test 2: All monitors report safe
result = decision_unit.update(
    perclos_result={"status": "SAFE"},
    head_pose_result={"status": "SAFE"},
    action_result={"status": "SAFE"},
    mouth_result={"status": "SAFE"},
    tracking_valid=True,
    calibrated=True,
)
print("Test 2 - All safe:", result["status"])

# Test 3: Head pose reports low awareness
result = decision_unit.update(
    perclos_result={"status": "SAFE"},
    head_pose_result={"status": "LOW_AWARENESS"},
    action_result={"status": "SAFE"},
    mouth_result={"status": "SAFE"},
    tracking_valid=True,
    calibrated=True,
)
print("Test 3 - Head pose risk:", result["status"])

# Test 4: Tracking is invalid
result = decision_unit.update(
    perclos_result={"status": "SAFE"},
    head_pose_result={"status": "SAFE"},
    action_result={"status": "SAFE"},
    mouth_result={"status": "SAFE"},
    tracking_valid=False,
    calibrated=True,
)
print("Test 4 - Invalid tracking:", result["status"])