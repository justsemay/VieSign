from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Đổi tên file model đúng với file .keras của bạn
MODEL_FILENAME = "cnn_bilstm_word_standard_24052026_03.keras"

# Có 2 kiểu file norm trong code cũ của bạn:
# 1) norm_stats.npz
# 2) <model_name>_norm_stats.npz
# Code sẽ ưu tiên NORM_FILENAME, nếu không có sẽ tự tìm *_norm_stats.npz
NORM_FILENAME = "norm_stats.npz"
LABELS_FILENAME = "labels.json"
DISPLAY_LABELS_FILENAME = "display_labels.json"

SEQUENCE_LENGTH = 30
FEATURE_DIM = 225
THRESHOLD = 0.70
MOTION_THRESHOLD = 0.005
IDLE_RESET_AFTER = 20
PREDICTION_WINDOW = 10
MAJORITY_COUNT = 6

# Frontend gửi frame mỗi 120ms là tương đối nhẹ cho demo REST API.
# Nếu máy mạnh, có thể giảm xuống 80ms trong frontend/js/camera.js
