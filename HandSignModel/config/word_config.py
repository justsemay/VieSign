# Đường dẫn
import os

# BASE_DIR = HandSignModel/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn — tuyệt đối, chạy từ đâu cũng đúng
DATA_PATH  = os.path.join(BASE_DIR, "data", "word")
MODEL_DIR  = os.path.join(BASE_DIR, "models", "word")

# Tham số thu thập
NUM_SEQUENCES   = 160  # Số video/từ
SEQUENCE_LENGTH = 30   # Số frame/video

# Danh sách từ — thêm/bớt tại đây
ACTIONS = [
    # Giao tiếp
    "xin_chao", "xin_loi", "cam_on",

    # Xưng hô
    "toi", "ban", "thay", "co",

    # Động từ / hành động
    "muon", "hoc", "giup_do", "an", "uong", "trinh_bay", "lam_viec",

    # Trạng thái
    "vui", "buon", "khoe", "met",

    # Danh từ
    "du_an",

    # Không ký hiệu
    "IDLE"
]

# Tham số model
THRESHOLD = 0.60   # Ngưỡng confidence để chấp nhận kết quả