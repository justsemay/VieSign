from collections import deque
from typing import Dict, Any, Optional

import numpy as np


class RecognitionService:
    """Quản lý buffer 30 frame, motion detection, majority vote và câu kết quả."""

    def __init__(
        self,
        model_service,
        sequence_length: int = 30,
        motion_threshold: float = 0.005,
        idle_reset_after: int = 20,
        threshold: float = 0.70,
        prediction_window: int = 10,
        majority_count: int = 6,
    ):
        self.model_service = model_service
        self.sequence_length = sequence_length
        self.motion_threshold = motion_threshold
        self.idle_reset_after = idle_reset_after
        self.threshold = threshold
        self.majority_count = majority_count

        self.sequence = deque(maxlen=sequence_length)
        self.predictions = deque(maxlen=prediction_window)
        self.motion_buffer = deque(maxlen=10)
        self.sentence = []
        self.prev_keypoints: Optional[np.ndarray] = None
        self.idle_frames = 0
        self.is_signing = False
        self.current_result = None

    def reset(self) -> Dict[str, Any]:
        self.sequence.clear()
        self.predictions.clear()
        self.motion_buffer.clear()
        self.sentence.clear()
        self.prev_keypoints = None
        self.idle_frames = 0
        self.is_signing = False
        self.current_result = None
        return self._response(status="reset")

    def clear_sentence(self) -> Dict[str, Any]:
        self.sentence.clear()
        self.predictions.clear()
        self.sequence.clear()
        self.current_result = None
        return self._response(status="cleared")

    def remove_last_word(self) -> Dict[str, Any]:
        """Xóa từ cuối cùng trong câu nhận dạng.

        Chỉ xóa dữ liệu đầu ra, không reset toàn bộ camera/model. Sau khi xóa,
        predictions được clear để model không thêm lại ngay từ vừa xóa do majority vote còn lưu nhãn cũ.
        """
        removed_word = None
        removed_word_display = None
        if self.sentence:
            removed_word = self.sentence.pop()
            removed_word_display = self.model_service.display_text(removed_word)

        self.predictions.clear()
        return_response = self._response(status="backspace")
        return_response.update({
            "removed_word": removed_word,
            "removed_word_display": removed_word_display,
        })
        return return_response

    def process_keypoints(self, keypoints: np.ndarray, meta: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        keypoints = np.asarray(keypoints, dtype="float32")

        avg_motion = 0.0
        if self.prev_keypoints is not None:
            # Từ index 99 trở đi là landmark tay trái + tay phải theo pipeline cũ
            hand_motion = float(np.mean(np.abs(keypoints[99:] - self.prev_keypoints[99:])))
            self.motion_buffer.append(hand_motion)
            avg_motion = float(np.mean(self.motion_buffer))
        self.prev_keypoints = keypoints.copy()

        if avg_motion > self.motion_threshold:
            self.is_signing = True
            self.idle_frames = 0
        else:
            if self.is_signing:
                self.idle_frames += 1
                if self.idle_frames >= self.idle_reset_after:
                    self.is_signing = False
                    self.idle_frames = 0
                    self.sequence.clear()
                    self.predictions.clear()
                    self.current_result = None

        self.sequence.append(keypoints)

        added_word = None
        added_word_display = None
        if len(self.sequence) == self.sequence_length and self.is_signing:
            result = self.model_service.predict(np.array(self.sequence, dtype="float32"))
            self.current_result = result
            pred_label = result["label"]
            pred_conf = result["confidence"]
            self.predictions.append(pred_label)

            if pred_label != "IDLE" and pred_conf > self.threshold:
                if self.predictions.count(pred_label) >= self.majority_count:
                    if not self.sentence or self.sentence[-1] != pred_label:
                        self.sentence.append(pred_label)
                        added_word = pred_label
                        added_word_display = self.model_service.display_text(pred_label)

        response = self._response(status="ok")
        response.update({
            "motion": avg_motion,
            "added_word": added_word,
            "added_word_display": added_word_display,
            "meta": meta or {},
        })
        return response

    def _sentence_display(self):
        return [self.model_service.display_text(label) for label in self.sentence]

    def _sentence_text(self) -> str:
        # Hiện tại hệ thống nhận dạng ở mức từ/cụm từ đơn, vì vậy cách nối an toàn nhất là dùng khoảng trắng.
        # Dấu câu chưa tự động thêm vì model chưa học ngữ cảnh câu.
        return " ".join(self._sentence_display()).strip()

    def _response(self, status: str = "ok") -> Dict[str, Any]:
        current_label = None
        current_display = None
        confidence = 0.0
        top3 = []
        if self.current_result:
            current_label = self.current_result["label"]
            current_display = self.current_result.get("display") or self.model_service.display_text(current_label)
            confidence = self.current_result["confidence"]
            top3 = self.current_result["top3"]

        sentence_display = self._sentence_display()

        return {
            "status": status,
            "is_signing": self.is_signing,
            "buffer": len(self.sequence),
            "sequence_length": self.sequence_length,
            "current_label": current_label,
            "current_display": current_display,
            "confidence": confidence,
            "top3": top3,
            "sentence": self.sentence,
            "sentence_display": sentence_display,
            "sentence_text": self._sentence_text(),
        }
