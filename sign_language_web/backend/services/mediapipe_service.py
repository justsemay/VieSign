from typing import Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np


class MediaPipeService:
    """Xử lý MediaPipe Holistic và trích xuất vector 225 đặc trưng."""

    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        image_rgb.flags.writeable = True

        keypoints = self.extract_keypoints(results)
        meta = {
            "has_pose": results.pose_landmarks is not None,
            "has_left_hand": results.left_hand_landmarks is not None,
            "has_right_hand": results.right_hand_landmarks is not None,
        }
        return keypoints, meta

    @staticmethod
    def extract_keypoints(results) -> np.ndarray:
        pose = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
            if results.pose_landmarks
            else np.zeros(33 * 3)
        )
        lh = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            if results.left_hand_landmarks
            else np.zeros(21 * 3)
        )
        rh = (
            np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            if results.right_hand_landmarks
            else np.zeros(21 * 3)
        )
        return np.concatenate([pose, lh, rh]).astype("float32")  # shape: (225,)
