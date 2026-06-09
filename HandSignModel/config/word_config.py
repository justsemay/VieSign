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
    "toi", "ban", "thay", "co", "bo", "me", "hang_xom",

    # Động từ / hành động
    "muon", "hoc", "giup_do", "an", "uong", "trinh_bay", "lam_viec",
    "tra_loi", "viet", "di", "xem", "doc_sach", "nghi_ngoi",

    # Trạng thái / cảm xúc / nhu cầu
    "vui", "buon", "khoe", "met", "khoc", "doi", "khat",
    "nong", "lanh", "gian_du", "nguy_hiem",

    # Danh từ / địa điểm / vật dụng
    "du_an", "da_bong", "nha_ve_sinh", "lop", "truong",
    "dien_thoai", "tien", "phim",

    # Phương tiện
    "o_to", "xe_dap", "xe_may",

    # Thời gian
    "buoi_sang", "buoi_trua", "buoi_toi",

    # Nhận thức
    "hieu",

    # Không ký hiệu
    "IDLE"
]

# Tham số model
THRESHOLD = 0.60   # Ngưỡng confidence để chấp nhận kết quả