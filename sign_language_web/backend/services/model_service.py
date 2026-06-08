import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from tensorflow.keras.models import load_model


class ModelService:
    def __init__(
        self,
        model_dir: Path,
        model_filename: str,
        norm_filename: str,
        labels_filename: str,
        display_labels_filename: str = "display_labels.json",
    ):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / model_filename
        self.norm_path = self._resolve_norm_path(norm_filename)
        self.labels_path = self.model_dir / labels_filename
        self.display_labels_path = self.model_dir / display_labels_filename

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {self.model_path}\n"
                f"Hãy copy file .keras vào backend/models hoặc sửa MODEL_FILENAME trong backend/config.py"
            )
        if not self.norm_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy norm stats: {self.norm_path}\n"
                f"Hãy copy norm_stats.npz vào backend/models"
            )
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Không tìm thấy labels.json: {self.labels_path}")

        self.labels = self._load_labels()
        self.display_labels = self._load_display_labels()
        self.model = load_model(self.model_path)

        norm = np.load(self.norm_path)
        self.mean = norm["mean"].astype("float32")
        self.std = norm["std"].astype("float32")
        self.std = np.where(self.std == 0, 1e-6, self.std)

        output_classes = int(self.model.output_shape[-1])
        if output_classes != len(self.labels):
            raise ValueError(
                f"Số nhãn trong labels.json ({len(self.labels)}) khác output model ({output_classes}).\n"
                f"Hãy kiểm tra thứ tự và số lượng ACTIONS."
            )

        print("✅ Model loaded")
        print(f"   Model: {self.model_path}")
        print(f"   Norm : {self.norm_path}")
        print(f"   Input: {self.model.input_shape}")
        print(f"   Labels: {len(self.labels)}")

    def _resolve_norm_path(self, norm_filename: str) -> Path:
        direct = self.model_dir / norm_filename
        if direct.exists():
            return direct
        candidates = sorted(self.model_dir.glob("*_norm_stats.npz"))
        if candidates:
            return candidates[0]
        return direct

    def _load_labels(self) -> List[str]:
        with open(self.labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if not isinstance(labels, list) or not labels:
            raise ValueError("labels.json phải là một list nhãn")
        return labels

    def _load_display_labels(self) -> Dict[str, str]:
        """
        labels.json giữ nhãn không dấu để khớp model.
        display_labels.json dùng riêng cho giao diện và TTS tiếng Việt có dấu.
        """
        if not self.display_labels_path.exists():
            return {label: self.format_label(label) for label in self.labels}

        with open(self.display_labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("display_labels.json phải là object dạng {raw_label: text_có_dấu}")

        # Nếu thiếu nhãn nào thì fallback sang format cơ bản, tránh làm server chết vì thiếu 1 key.
        return {label: data.get(label, self.format_label(label)) for label in self.labels}

    def format_label(self, label: str) -> str:
        return str(label).replace("_", " ")

    def display_text(self, label: str) -> str:
        return self.display_labels.get(label, self.format_label(label))

    def predict(self, sequence: np.ndarray) -> Dict[str, Any]:
        """sequence shape: (30, 225)"""
        sequence = np.asarray(sequence, dtype="float32")
        if sequence.ndim != 2:
            raise ValueError(f"Sequence phải có shape (T, F), hiện tại: {sequence.shape}")

        input_data = np.expand_dims(sequence, axis=0).astype("float32")
        input_data = (input_data - self.mean) / self.std

        probs = self.model.predict(input_data, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = self.labels[pred_idx]
        pred_conf = float(probs[pred_idx])

        top3_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {
                "label": self.labels[int(i)],
                "display": self.display_text(self.labels[int(i)]),
                "confidence": float(probs[int(i)]),
            }
            for i in top3_idx
        ]

        return {
            "label": pred_label,
            "display": self.display_text(pred_label),
            "confidence": pred_conf,
            "top3": top3,
            "raw": probs.tolist(),
        }
