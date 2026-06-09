import base64
import re
from typing import Tuple

import cv2
import numpy as np


def decode_base64_image(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("Không có dữ liệu ảnh")

    # Chấp nhận cả dạng data:image/jpeg;base64,... và raw base64
    data = re.sub(r"^data:image/.+;base64,", "", data_url)
    img_bytes = base64.b64decode(data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Không decode được ảnh base64")

    return frame


def resize_keep_ratio(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)))
