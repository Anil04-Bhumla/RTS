# -*- coding: utf-8 -*-
# ---------------------

import numpy as np

from .dms.dms_inference import DMSInfer
from .face_detector.detector import Detector as FaceDetector
from .dms.utils import principal_roi, iou_of
from .third_party import KalmanBoxTracker

class DriverTacker:

    def __init__(self, fd_model, dms_model, start_n, reset_n, min_iou=0.6, device='cuda:0'):
        """
        :param fd_model: face detection onnx model name
        :param dms_model: dmsNet onnx model name
        :param start_n: Number of consecutive valid detections before start tracking
        :param reset_n: Number of consecutive invalid detections to loose tracking
        :param min_iou: Minimum iou to associate two consecutive BB's
        :param device: Device for pytorch inference (deprecated)
        """

        # Create inference classes instances
        self.__face_detector = FaceDetector(fd_model, device=device)
        self.__dms_infer = DMSInfer(dms_model, device=device)
        self.inference_latency_ms = 0.0

        # Configurable options
        self.start_after = start_n  # Set after N consecutive valid detection
        self.reset_after = reset_n  # Reset after N frames without any detection
        self.min_iou = min_iou

        # Default options
        self.id_list = []
        self.id_consec_detects = 3

        self.is_tracking = False
        self.is_last_valid = False
        self.consec_invalid_frames = 0
        self.valid_frames = 0
        self.last_valid_roi = None

        # KF box tracker
        self.KF = None
        self.trk_frames = 0


    def update(self, frame):

        if self.is_tracking:
            # predict
            self.is_last_valid = True
            self.last_valid_roi = self.KF.predict()[0]
            self.trk_frames += 1

        if not self.is_tracking or (self.is_tracking and self.trk_frames % 1 == 0):
            # update
            self.detect(frame)

        if self.is_tracking and self.is_last_valid:
            # Perform DMS_NET inference...
            lmks, euler, eyes, mouth_s, action = self.__dms_infer.detect(frame, self.last_valid_roi)
            self.inference_latency_ms = self.__dms_infer.inference_latency_ms

            return self.is_tracking, lmks, euler, eyes, mouth_s, action

        else:
            return self.is_tracking, np.array([]), np.array([0,0,0]), np.array([]), 0, 0

    def detect(self, frame):
        # 1: Detect all faces in frame
        bboxes, labels = self.__face_detector.detect(frame)
        bboxes = bboxes[labels == 1]

        if len(bboxes) != 0:

            box_id = 0
            # Find principal (i.e bigger) ROI
            if len(bboxes) > 1:
                box_id = principal_roi(bboxes)

            # If only a single bb is detected then it is automatically the biggest
            if self.last_valid_roi is not None:
                iou = iou_of(bboxes[box_id], self.last_valid_roi)
                if iou < 0.6:
                    self._update_status(False)
                else:
                    bbox = bboxes[box_id]
                    bbox = bbox.astype(int)
                    self._update_status(True, bbox)

            else:
                bbox = bboxes[box_id]
                bbox = bbox.astype(int)
                self._update_status(True, bbox)

        else:
            self._update_status(False)

    def _update_status(self, is_ok, bbox=None):

        self.is_last_valid = is_ok
        if not is_ok:
            self.consec_invalid_frames += 1
            if self.consec_invalid_frames > self.reset_after:
                # stop tracking
                # print("STOP")
                self.consec_invalid_frames = 0
                self.is_tracking = False
                self.last_valid_roi = None
                self.valid_frames = 0
                self.KF = None
                self.trk_frames = 0

        else:
            self.valid_frames += 1
            self.consec_invalid_frames = 0
            if not self.is_tracking:
                self.last_valid_roi = bbox
                if self.valid_frames > self.start_after:
                    # print("START")
                    # start tracking
                    self.is_tracking = True
                    self.KF = KalmanBoxTracker(bbox)

            if self.is_tracking:
                self.KF.update(bbox)
