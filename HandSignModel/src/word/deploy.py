import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from collections import deque
import os, sys
import glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import ACTIONS, MODEL_DIR, SEQUENCE_LENGTH, THRESHOLD

# ── Tuning params ── THÊM VÀO ĐÂY ──────────────────────────
MOTION_THRESHOLD = 0.005   # Điều chỉnh sau khi chạy thử
IDLE_RESET_AFTER = 20      # ~0.7s ở 30fps

# ── Setup MediaPipe ──────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_draw     = mp.solutions.drawing_utils

def extract_keypoints(results):
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 3)
    lh   = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
           if results.left_hand_landmarks else np.zeros(21 * 3)
    rh   = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
           if results.right_hand_landmarks else np.zeros(21 * 3)
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

def draw_confidence_bars(frame, res):
    """Vẽ thanh confidence cho từng từ"""
    for i, (action, conf) in enumerate(zip(ACTIONS, res)):
        bar_w = int(conf * 200)
        color = (0, 255, 0) if conf == res.max() else (100, 100, 100)
        y = 100 + i * 35

        # Background bar
        cv2.rectangle(frame, (frame.shape[1]-220, y),
                      (frame.shape[1]-20, y+25), (50,50,50), -1)
        # Confidence bar
        if bar_w > 0:
            cv2.rectangle(frame, (frame.shape[1]-220, y),
                          (frame.shape[1]-220+bar_w, y+25), color, -1)
        # Label
        cv2.putText(frame, f"{action}: {conf*100:.1f}%",
            (frame.shape[1]-218, y+18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)


# ── Load model ───────────────────────────────────────────────
MODEL_FILENAME = "cnn_bilstm_word_standard_24052026_03.keras"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
print(f"📂 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print(f"✅ Model loaded | Input: {model.input_shape} | Classes: {len(ACTIONS)}")
norm_path = os.path.join(MODEL_DIR, "norm_stats.npz")

if not os.path.exists(norm_path):
    raise FileNotFoundError(f"❌ Không tìm thấy norm_stats.npz: {norm_path}")

norm = np.load(norm_path)
mean = norm["mean"]
std = norm["std"]

print("✅ Loaded norm_stats.npz")
# ── State variables ──────────────────────────────────────────
sequence    = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=10)
sentence    = []       # Danh sách từ đã nhận dạng
current_res = None     # Kết quả predict hiện tại
prev_keypoints = None
motion_buffer  = deque(maxlen=10)
idle_frames    = 0
is_signing     = False
# ── Main loop ────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🚀 Bắt đầu nhận dạng | Q = thoát | C = xóa câu\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ── MediaPipe process ────────────────────────────────
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ── Draw landmarks ───────────────────────────────────
        draw_landmarks(frame, results)

        # ── Extract & buffer keypoints ───────────────────────
        keypoints = extract_keypoints(results)
        if prev_keypoints is not None:
            hand_motion = np.mean(np.abs(keypoints[99:] - prev_keypoints[99:]))
            motion_buffer.append(hand_motion)
            avg_motion = np.mean(motion_buffer)
        else:
            avg_motion = 0.0
        prev_keypoints = keypoints.copy()

        # Cập nhật trạng thái
        if avg_motion > MOTION_THRESHOLD:
            is_signing  = True
            idle_frames = 0
        else:
            if is_signing:
                idle_frames += 1
                if idle_frames >= IDLE_RESET_AFTER:
                    is_signing  = False
                    idle_frames = 0
                    sequence.clear()
                    predictions.clear()
                    current_res = None
        sequence.append(keypoints)

        # ── Progress bar ─────────────────────────────────────
        progress = int(len(sequence) / SEQUENCE_LENGTH * 400)
        cv2.rectangle(frame,
            (20, frame.shape[0]-50),
            (420, frame.shape[0]-25), (50,50,50), -1)
        bar_color = (0,255,0) if is_signing else (100,100,100)  # ← xanh/xám theo trạng thái
        cv2.rectangle(frame,
            (20, frame.shape[0]-50),
            (20 + progress, frame.shape[0]-25), bar_color, -1)
        cv2.putText(frame,
            f"{'[KY HIEU]' if is_signing else '[CHO...]'}  Motion:{avg_motion:.4f}  Buffer:{len(sequence)}/{SEQUENCE_LENGTH}",
            (20, frame.shape[0]-55),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        # ── Predict khi đủ frame ─────────────────────────────
        if len(sequence) == SEQUENCE_LENGTH and is_signing:
            input_data = np.expand_dims(list(sequence), axis=0).astype("float32")
            input_data = (input_data - mean) / std

            current_res = model.predict(input_data, verbose=0)[0]


            top3_idx = np.argsort(current_res)[::-1][:3]
            print(f"Top3: " + " | ".join(
            [f"{ACTIONS[i]}({current_res[i]*100:.0f}%)" for i in top3_idx]
            ))
            pred_idx   = np.argmax(current_res)
            pred_conf  = current_res[pred_idx]
            pred_label = ACTIONS[pred_idx]

            predictions.append(pred_label)

            # Stable prediction: majority vote trong 10 frame gần nhất
            if pred_label == "IDLE":
                pass   # ← Bước 1: Bỏ qua IDLE, không thêm vào sentence

            elif pred_conf > THRESHOLD:
                if predictions.count(pred_label) >= 6:
                    if not sentence or sentence[-1] != pred_label:
                        sentence.append(pred_label)
                        print(f"  🟢 Nhận dạng: [{pred_label}] ({pred_conf*100:.1f}%)")

        # ── Vẽ confidence bars ───────────────────────────────
        if current_res is not None:
            draw_confidence_bars(frame, current_res)

        # ── Vẽ sentence (các từ đã nhận dạng) ───────────────
        # Panel đen phía trên
        cv2.rectangle(frame, (0,0), (frame.shape[1], 70), (0,0,0), -1)

        # Hiển thị các từ
        display = " → ".join(sentence[-4:]) if sentence else "Dang cho ky hieu..."
        cv2.putText(frame, display,
            (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
            (255,255,255) if sentence else (150,150,150), 2)

        # Hướng dẫn phím
        cv2.putText(frame, "Q: Thoat  |  C: Xoa  |  S: Luu",
            (20, frame.shape[0]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        cv2.imshow("VSL Word Recognition", frame)

        # ── Xử lý phím ──────────────────────────────────────
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):
            print("\n⛔ Thoát!")
            break

        elif key == ord('c'):
            sentence.clear()
            predictions.clear()
            sequence.clear()
            current_res = None
            print("  🗑️  Đã xóa câu")

        elif key == ord('s'):
            result_str = " ".join(sentence)
            print(f"  💾 Lưu: '{result_str}'")
            cv2.imwrite(f"./output_{len(sentence)}.jpg", frame)

cap.release()
cv2.destroyAllWindows()
print("✅ Kết thúc!")