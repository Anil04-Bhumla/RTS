'''
Copyright (C) University of Modena and Reggio Emilia - All Rights Reserved
Unauthorized copying of this file, via any medium is strictly prohibited
Proprietary and confidential
Written by Carmelo Scribano <carmelo.scribano@unimore.it>
'''

import cv2
import numpy as np

from src import DriverTacker

def main():

    # Video Capture
    cap = cv2.VideoCapture(0)

    # Inferdriver_trackerence configuration
    face_detector = "version-RFB-320_sim.onnx"
    dms_model = "reborn.onnx"

    # Create tracker class instance
    driver_tracker = DriverTacker(face_detector, dms_model, 5, 20, 0, device='cpu')

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        is_traking, lmks, euler, eyes, mouth_s, action = driver_tracker.update(frame)
        print(euler)

        # Display
        if lmks is not None and len(lmks) > 0:
            points = np.array(lmks, dtype=int).reshape(-1, 2)
            for point in points:
                frame = cv2.circle(frame, tuple(point), 2, (255, 229, 10), -1, 1)

        cv2.imshow("result", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
