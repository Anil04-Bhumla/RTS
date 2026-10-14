'''
Copyright (C) University of Modena and Reggio Emilia - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Carmelo Scribano <carmelo.scribano@unimore.it>
'''

from src.dms import driver_calibration
from src import driver_tracker
import cv2
import numpy as np
import time 
import csv
from datetime import datetime
from src.dms.perclos_monitor import PerclosMonitor
from src.dms.alert_decision import AlertDecision
from src.dms.driver_calibration import DriverCalibration
from collections import deque

from src import DriverTacker
from src.dms.head_pose_monitor import HeadPoseMonitor
from src.dms.action_monitor import ActionMonitor
from src.dms.yawning_monitor import YawningMonitor
from src.dms.decision_unit import DecisionUnit

def main():

    # Video Capture
    cap = cv2.VideoCapture(0)


    perclos_monitor = PerclosMonitor(   

        window_seconds=60.0,

        opening_threshold=0.20,

        visibility_threshold=0.50,

        max_sample_gap_seconds=0.25,

    )
    alert_decision = AlertDecision(
        warning_threshold=15.0,
        danger_threshold=25.0,
    )
    driver_calibration = DriverCalibration(required_samples=60)
    head_pose_monitor = HeadPoseMonitor(
        yaw_threshold=20.0,
        pitch_threshold=15.0,
        roll_threshold=15.0,
        sustained_duration=2.0,
    )
    action_monitor = ActionMonitor(
        confirmation_duration=1.0,
        history_duration=2.0,
        minimum_detection_ratio=0.7,
    )
    yawning_monitor = YawningMonitor(
        confirmation_duration=2.0,
    )
    decision_unit = DecisionUnit(cooling_period=2.0)

    # Real-time performance monitoring

    latency_history = deque(maxlen=10000)
    deadline_ms = 100.0  # Initial experimental deadline: 100 ms
    deadline_misses = 0
    total_frames = 0

    # CSV performance logging
    log_filename = (
        f"rts_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )   

    log_file = open(
        log_filename,
        mode="w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(log_file)

    csv_writer.writerow([
        "timestamp",
        "frame_number",
        "tracker_latency_ms",
        "fps",
        "deadline_ms",
        "deadline_miss",
        "eye_state",
        "perclos_percent",
        "combined_status",
    ])

    prev_time = time.time()

    # Inferdriver_trackerence configuration
    face_detector = "face_detect/version-RFB-320_sim.onnx"
    dms_model = "dms/reborn.onnx"

    # Create tracker class instance
    driver_tracker = DriverTacker(face_detector, dms_model, 5, 20, 0, device='cpu')

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = current_time
        
        # Real-time performance measurement
        frame_start_time = time.perf_counter()

        # Inference
        is_traking, lmks, euler, eyes, mouth_s, action = driver_tracker.update(frame)
        tracking_valid = (
            bool(is_traking)
            and lmks is not None
            and len(lmks) > 0
        )

        calibration_result = driver_calibration.update(
            euler,
            tracking_valid=tracking_valid,
        )

        head_pose_deviation = None

        if tracking_valid and driver_calibration.is_calibrated:
            head_pose_deviation = (
                driver_calibration.get_head_pose_deviation(euler)
            )

        head_pose_result = head_pose_monitor.update(
            head_pose_deviation,
            tracking_valid=(
                tracking_valid
                and driver_calibration.is_calibrated
            ),
        )
        action_result = action_monitor.update(
            int(action),
            tracking_valid=tracking_valid,
        )
        yawning_result = yawning_monitor.update(
            int(mouth_s),
            tracking_valid=tracking_valid,
        )



        

        frame_end_time = time.perf_counter()

        # Total tracker processing latency in milliseconds
        frame_latency_ms = (
            frame_end_time - frame_start_time
        ) * 1000.0

        latency_history.append(frame_latency_ms)
        total_frames += 1

        # Count frames that exceed the configured deadline
        if frame_latency_ms > deadline_ms:
            deadline_misses += 1
        # PERCLOS calculation


        perclos_result = perclos_monitor.update(
            eyes,
            tracking_valid=tracking_valid,
        )

        eye_state = perclos_result["eye_state"]
        perclos_value = perclos_result["perclos"]
        alert_result = alert_decision.update(
            perclos=perclos_value,
            eye_state=eye_state,
        )
        
        alert_status = alert_result["status"]
        alert_message = alert_result["message"]


        combined_result = decision_unit.update(
            perclos_result=alert_result,
            head_pose_result=head_pose_result,
            action_result=action_result,
            mouth_result=yawning_result,
            tracking_valid=tracking_valid,
            calibrated=driver_calibration.is_calibrated,
        )
        combined_status = combined_result["status"]
        combined_message = combined_result["message"]



        
        inference_latency = driver_tracker.inference_latency_ms

        # Calculate real-time performance metrics
        if latency_history:
            average_latency_ms = float(np.mean(latency_history))
            worst_case_latency_ms = float(np.max(latency_history))
            deadline_miss_percentage = (
                deadline_misses / total_frames
            ) * 100.0
        else:
            average_latency_ms = 0.0
            worst_case_latency_ms = 0.0
            deadline_miss_percentage = 0.0

        # Save current frame metrics to CSV
        csv_writer.writerow([
            datetime.now().isoformat(timespec="milliseconds"),
            total_frames,
            round(frame_latency_ms, 2),
            round(fps, 2),
            deadline_ms,
            int(frame_latency_ms > deadline_ms),
            eye_state,
            round(perclos_value, 2) if perclos_value is not None else "",
            combined_status,
        ])

        # Ensure data is written regularly
        log_file.flush()

        

        cv2.putText(
            frame,
            f"Inference Latency: {inference_latency:.2f} ms",
            (20, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        print(euler)
        print("Eyes:", eyes)
        print("Mouth:", mouth_s)
        print("Action:", action)
        cv2.putText(frame, f"Yaw: {euler[0]:.1f}", (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Pitch: {euler[1]:.1f}", (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Roll: {euler[2]:.1f}", (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Eyes: {eyes}", (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # Display eye state
        cv2.putText(
            frame,
            f"Eye State: {eye_state}",
            (30, 575),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # Display PERCLOS
        if perclos_value is not None:
            perclos_text = f"PERCLOS (60s): {perclos_value:.1f}%"
        else:
            perclos_text = "PERCLOS (60s): Collecting..."

        cv2.putText(
            frame,
            perclos_text,
            (20, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        # Display driver alert status
        alert_colors = {
            "CALIBRATION": (0, 255, 255),
            "SAFE": (0, 255, 0),
            "WARNING": (0, 255, 255),
            "LOW_AWARENESS": (0, 165, 255),
            "DANGER": (0, 0, 255),
            "UNKNOWN": (200, 200, 200),
        }

        alert_color = alert_colors.get(
            combined_status,
            (200, 200, 200),
        )
        cv2.putText(
            frame,
            f"Driver Status: {combined_status}",
            (20, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            alert_color,
            2,
        )

        if not driver_calibration.is_calibrated:
            calibration_text = (
                f"Calibration: {calibration_result['progress']:.0f}%"
            )
            calibration_color = (0, 255, 255)

        else:
            baseline = driver_calibration.baseline_head_pose

            calibration_text = (
                f"Baseline Y:{baseline[0]:.1f} "
                f"P:{baseline[1]:.1f} "
                f"R:{baseline[2]:.1f}"
            )
            calibration_color = (0, 255, 0)

        cv2.putText(
            frame,
            calibration_text,
            (20, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            calibration_color,
            2,
        )
        # Display head-pose deviation
        if head_pose_deviation is not None:
            deviation_text = (
                f"Head Deviation Y:{head_pose_deviation[0]:.1f} "
                f"P:{head_pose_deviation[1]:.1f} "
                f"R:{head_pose_deviation[2]:.1f}"
            )
        else:
            deviation_text = "Head Deviation: Calibrating"

        cv2.putText(
            frame,
            deviation_text,
            (30, 610),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Display head-pose monitoring status
        head_status = head_pose_result["status"]
        head_duration = head_pose_result["duration"]

        head_status_text = (
            f"Head Status: {head_status} "
            f"({head_duration:.1f}s)"
        )

        if head_status == "LOW_AWARENESS":
            head_status_color = (0, 0, 255)       # Red
        elif head_status == "MONITORING":
            head_status_color = (0, 255, 255)     # Yellow
        elif head_status == "SAFE":
            head_status_color = (0, 255, 0)        # Green
        else:
            head_status_color = (255, 255, 255)    # White

        cv2.putText(
            frame,
            head_status_text,
            (30, 645),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            head_status_color,
            2,
        )




        cv2.putText(
            frame,
            alert_message,
            (20, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            alert_color,
            1,
        )
        # Convert mouth class index into a readable label
        mouth_labels = {
            0: "Closed",
            1: "Semi-open",
            2: "Open"
        }

        mouth_text = mouth_labels.get(int(mouth_s), "Unknown")




        # Convert distraction class index into a readable label
        action_labels = {
            0: "Normal",
            1: "Phone Use",
            2: "Smoking"
        }

        action_text = action_labels.get(int(action), "Unknown")

        action_status = action_result["status"]
        monitored_action = action_result["action"]
        action_duration = action_result["duration"]

        # Display mouth and distraction labels
        cv2.putText(frame, f"Mouth: {mouth_text}", (20, 210),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"Distraction: {action_text}", (20, 240),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(
            frame,
            f"Avg Latency: {average_latency_ms:.2f} ms",
            (20, 310),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Action Monitor: {action_status}",
            (20, 535),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        if monitored_action is not None:
            cv2.putText(
                frame,
                f"Action: {monitored_action} ({action_duration:.1f}s)",
                (20, 560),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Worst Latency: {worst_case_latency_ms:.2f} ms",
            (20, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        # Display sustained mouth-opening status
        yawning_status = yawning_result["status"]
        yawning_duration = yawning_result["duration"]

        yawning_text = (
            f"Mouth Monitor: {yawning_status} "
            f"({yawning_duration:.1f}s)"
        )
        if yawning_status == "DETECTED":
            yawning_color = (0, 0, 255)
        elif yawning_status == "MONITORING":
            yawning_color = (0, 255, 255)
        elif yawning_status == "SAFE":
            yawning_color = (0, 255, 0)
        else:
            yawning_color = (255, 255, 255)

        cv2.putText(
            frame,
            yawning_text,
            (20, 680),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            yawning_color,
            2,
        )

        cv2.putText(
            frame,
            f"Deadline Miss: {deadline_miss_percentage:.2f}%",
            (20, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255) if deadline_miss_percentage > 0 else (0, 255, 0),
            2,
        )

        

        # Display
        if lmks is not None and len(lmks) > 0:
            points = np.array(lmks, dtype=int).reshape(-1, 2)
            for point in points:
                frame = cv2.circle(frame, tuple(point), 2, (255, 229, 10), -1, 1)

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),

            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("result", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release resources and close the CSV file
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

    print(f"Performance log saved to: {log_filename}")
    print(f"Total frames recorded: {total_frames}")
    print(f"Deadline misses: {deadline_misses}")

if __name__ == '__main__':
    main()

