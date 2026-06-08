# src/word/collect_idle.py
import cv2
import numpy as np
import mediapipe as mp
import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import DATA_PATH, SEQUENCE_LENGTH

mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

# ── Config ───────────────────────────────────────────────────
IDLE_CLASS    = "IDLE"
NUM_SEQUENCES = 60     # Thu 60 video cho class IDLE
DATA_DIR      = os.path.join(DATA_PATH, IDLE_CLASS)

def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33*3)
    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21*3)
    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

def draw_landmarks(frame, results):
    mp_draw.draw_landmarks(frame, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(80,110,10),  thickness=1, circle_radius=1),
        mp_draw.DrawingSpec(color=(80,256,121), thickness=1))
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(121,22,76),  thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(121,44,250), thickness=2))
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(245,66,230), thickness=2))

# ── Tạo thư mục ──────────────────────────────────────────────
for seq in range(NUM_SEQUENCES):
    os.makedirs(os.path.join(DATA_DIR, str(seq)), exist_ok=True)

print(f"📂 Sẽ lưu vào: {DATA_DIR}")
print(f"📊 Cần thu: {NUM_SEQUENCES} video x {SEQUENCE_LENGTH} frame\n")

# ── Hướng dẫn ────────────────────────────────────────────────
IDLE_INSTRUCTIONS = [
    ("Tay tha xuong tu nhien, dung yen",    range(0,  20)),   # 20 video đứng im
    ("Di chuyen tay ngau nhien (khong ky)", range(20, 40)),   # 20 video chuyển tiếp
    ("Lam gi do tu nhien (gai dau, etc.)",  range(40, 60)),   # 20 video cử động ngẫu nhiên
]

def get_instruction(seq_idx):
    for text, r in IDLE_INSTRUCTIONS:
        if seq_idx in r:
            return text
    return "Thu thap IDLE"

# ── Main ─────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    seq = 0   # Bắt đầu từ video 0

    # Kiểm tra đã có data chưa để tiếp tục
    for s in range(NUM_SEQUENCES):
        check = os.path.join(DATA_DIR, str(s), "0.npy")
        if not os.path.exists(check):
            seq = s
            break
    print(f"▶ Bắt đầu từ video #{seq}\n")

    while seq < NUM_SEQUENCES and cap.isOpened():
        instruction = get_instruction(seq)

        # ── Màn hình chờ ─────────────────────────────────────
        print(f"\n[{seq+1}/{NUM_SEQUENCES}] {instruction}")
        print("  Nhấn SPACE để bắt đầu thu | Q để thoát")

        while True:
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_landmarks(frame, results)

            # UI màn hình chờ
            cv2.rectangle(frame, (0,0), (frame.shape[1], 120), (0,0,0), -1)
            cv2.putText(frame, f"IDLE [{seq+1}/{NUM_SEQUENCES}]",
                (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
            cv2.putText(frame, instruction,
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, "SPACE = bat dau  |  Q = thoat",
                (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            # Progress tổng
            done = sum(1 for s in range(NUM_SEQUENCES)
                       if os.path.exists(os.path.join(DATA_DIR, str(s), "0.npy")))
            cv2.putText(frame, f"Da thu: {done}/{NUM_SEQUENCES}",
                (frame.shape[1]-200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

            cv2.imshow("IDLE Data Collection", frame)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(' '):
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\n⛔ Thoát!")
                sys.exit()
        # ── Đếm ngược 3 giây trước khi thu ──────────────────
        for countdown in range(3, 0, -1):
            deadline = time.time() + 1.0   # Mỗi số hiện 1 giây

            while time.time() < deadline:
                ret, frame = cap.read()
                if not ret: break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                draw_landmarks(frame, results)

                # Nền tối
                cv2.rectangle(frame, (0,0), (frame.shape[1], 120), (0,0,0), -1)
                cv2.putText(frame, f"IDLE [{seq+1}/{NUM_SEQUENCES}]",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                cv2.putText(frame, instruction,
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # Số đếm ngược to ở giữa
                text_size = cv2.getTextSize(str(countdown),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
                cx = (frame.shape[1] - text_size[0]) // 2
                cy = (frame.shape[0] + text_size[1]) // 2
                cv2.putText(frame, str(countdown),
                    (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 6,
                    (0, 100, 255), 8)
                cv2.putText(frame, "Chuan bi...",
                    (cx - 30, cy + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                cv2.imshow("IDLE Data Collection", frame)
                cv2.waitKey(1)
        # ── Thu 30 frame ─────────────────────────────────────
        frame_num = 0
        while frame_num < SEQUENCE_LENGTH:
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            draw_landmarks(frame, results)

            # Lưu keypoints
            keypoints = extract_keypoints(results)
            save_path = os.path.join(DATA_DIR, str(seq), f"{frame_num}.npy")
            np.save(save_path, keypoints)

            # UI đang thu
            cv2.rectangle(frame, (0,0), (frame.shape[1], 120), (0,0,50), -1)
            cv2.putText(frame, f"🔴 DANG THU  Frame: {frame_num+1}/{SEQUENCE_LENGTH}",
                (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,100,255), 2)
            cv2.putText(frame, instruction,
                (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

            # Progress bar
            prog = int((frame_num+1) / SEQUENCE_LENGTH * 400)
            cv2.rectangle(frame, (15, 95), (415, 115), (50,50,50), -1)
            cv2.rectangle(frame, (15, 95), (15+prog, 115), (0,100,255), -1)

            cv2.imshow("IDLE Data Collection", frame)
            cv2.waitKey(1)
            frame_num += 1

        print(f"  ✅ Đã lưu video #{seq}")
        seq += 1
        time.sleep(0.3)   # Nghỉ nhỏ giữa các video

cap.release()
cv2.destroyAllWindows()

# ── Kiểm tra kết quả ─────────────────────────────────────────
total = sum(1 for s in range(NUM_SEQUENCES)
            if os.path.exists(os.path.join(DATA_DIR, str(s), "0.npy")))
print(f"\n✅ Hoàn thành! Đã thu {total}/{NUM_SEQUENCES} video cho class IDLE")
print(f"📂 Lưu tại: {DATA_DIR}")