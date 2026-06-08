import cv2
import json
import numpy as np
import mediapipe as mp
from collections import deque
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from PIL import Image, ImageDraw, ImageFont
# =========================================================
# CONFIG
# =========================================================

WEIGHTS_PATH = "vsl_201_cnn_bigru_attention_1.weights.h5"
CLASS_PATH = "labels.json"

SEQUENCE_LENGTH = 60
FEATURE_DIM = 1605

THRESHOLD = 0.85
PREDICT_EVERY = 3

MOTION_THRESHOLD = 0.002
IDLE_RESET_AFTER = 20

# =========================================================
# LOAD LABELS
# =========================================================

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    labels_dict = json.load(f)

class_names = list(labels_dict.values())
num_classes = len(class_names)

print("Classes:", num_classes)

# =========================================================
# BUILD MODEL + LOAD WEIGHTS
# =========================================================

model = Sequential([
    Bidirectional(
        LSTM(64, return_sequences=True),
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)
    ),
    Dropout(0.3),

    LSTM(64),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")
])

model.load_weights(WEIGHTS_PATH)

dummy = np.zeros((1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
model.predict(dummy, verbose=0)

print("✅ Model rebuilt + weights loaded")
print("Input shape:", model.input_shape)

# =========================================================
# MEDIAPIPE
# =========================================================

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# =========================================================
# EXTRACT KEYPOINTS
# =========================================================

def extract_keypoints(results):
    face = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
    ).flatten() if results.face_landmarks else np.zeros(468 * 3)

    lh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)

    pose = np.array(
        [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:25]]
    ).flatten() if results.pose_landmarks else np.zeros(25 * 3)

    keypoints = np.concatenate([face, lh, rh, pose])

    assert keypoints.shape[0] == FEATURE_DIM, keypoints.shape

    return keypoints.astype(np.float32)


def draw_landmarks(frame, results):
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def calc_motion(current_kp, prev_kp):
    if prev_kp is None:
        return 0.0

    hand_pose_start = 1404
    motion = np.mean(np.abs(current_kp[hand_pose_start:] - prev_kp[hand_pose_start:]))

    return float(motion)


def draw_vietnamese_text(frame, text, position, font_size=32, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_confidence_bars(frame, res, top_k=5):
    top_idx = np.argsort(res)[::-1][:top_k]

    x1 = frame.shape[1] - 330
    x2 = frame.shape[1] - 20

    for rank, idx in enumerate(top_idx):
        conf = float(res[idx])
        label = class_names[idx]

        y = 100 + rank * 35
        bar_w = int(conf * 220)

        cv2.rectangle(frame, (x1, y), (x2, y + 25), (50, 50, 50), -1)
        cv2.rectangle(frame, (x1, y), (x1 + bar_w, y + 25), (0, 255, 0), -1)

        text = f"{label}: {conf * 100:.1f}%"

        frame = draw_vietnamese_text(
            frame,
            text,
            (x1 + 5, y - 2),
            font_size=20,
            color=(255, 255, 255)
        )

    return frame



# =========================================================
# STATE
# =========================================================

sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=10)
sentence = []

current_res = None
prev_keypoints = None
motion_buffer = deque(maxlen=10)

idle_frames = 0
is_signing = False
frame_count = 0



# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n🚀 Bắt đầu nhận dạng | Q = thoát | C = xóa câu | S = lưu ảnh\n")

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        draw_landmarks(frame, results)

        keypoints = extract_keypoints(results)

        avg_motion = calc_motion(keypoints, prev_keypoints)
        motion_buffer.append(avg_motion)
        smooth_motion = float(np.mean(motion_buffer)) if motion_buffer else 0.0
        prev_keypoints = keypoints.copy()

        # =================================================
        # SIGNING STATE
        # =================================================

        if smooth_motion > MOTION_THRESHOLD:
            is_signing = True
            idle_frames = 0
        else:
            if is_signing:
                idle_frames += 1

                if idle_frames >= IDLE_RESET_AFTER:
                    is_signing = False
                    idle_frames = 0
                    sequence.clear()
                    predictions.clear()
                    current_res = None

        sequence.append(keypoints)

        # =================================================
        # PREDICT
        # =================================================

        if (
            len(sequence) == SEQUENCE_LENGTH
            and is_signing
            and frame_count % PREDICT_EVERY == 0
        ):
            input_data = np.expand_dims(np.array(sequence), axis=0)

            current_res = model.predict(input_data, verbose=0)[0]

            top3_idx = np.argsort(current_res)[::-1][:3]
            print(
                "Top3:",
                " | ".join([
                    f"{class_names[i]}({current_res[i] * 100:.1f}%)"
                    for i in top3_idx
                ])
            )

            pred_idx = int(np.argmax(current_res))
            pred_conf = float(current_res[pred_idx])
            pred_label = class_names[pred_idx]

            predictions.append(pred_label)

            if pred_conf >= THRESHOLD:
                if predictions.count(pred_label) >= 7:
                    if not sentence or sentence[-1] != pred_label:
                        sentence.append(pred_label)
                        print(f"🟢 Nhận dạng: {pred_label} ({pred_conf * 100:.1f}%)")

        # =================================================
        # UI
        # =================================================
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 75), (0, 0, 0), -1)

        display = " → ".join(sentence[-4:]) if sentence else "Đang chờ ký hiệu..."

        frame = draw_vietnamese_text(
            frame,
            display,
            (15, 18),
            font_size=34,
            color=(255, 255, 255) if sentence else (150, 150, 150)
        )

        progress = int(len(sequence) / SEQUENCE_LENGTH * 400)
        y_bar = frame.shape[0] - 50

        cv2.rectangle(frame, (20, y_bar), (420, y_bar + 25), (50, 50, 50), -1)

        bar_color = (0, 255, 0) if is_signing else (100, 100, 100)

        cv2.rectangle(frame, (20, y_bar), (20 + progress, y_bar + 25), bar_color, -1)

        state_text = "[KÝ HIỆU]" if is_signing else "[CHỜ...]"

        frame = draw_vietnamese_text(
            frame,
            f"{state_text} Motion:{smooth_motion:.4f} Buffer:{len(sequence)}/{SEQUENCE_LENGTH}",
            (20, y_bar - 25),
            font_size=18,
            color=(220, 220, 220)
        )

        if current_res is not None:
            frame = draw_confidence_bars(frame, current_res, top_k=5)

        frame = draw_vietnamese_text(
            frame,
            "Q: Thoát | C: Xóa | S: Lưu ảnh",
            (20, frame.shape[0] - 25),
            font_size=18,
            color=(180, 180, 180)
        )

        cv2.imshow("VSL Word Recognition", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == ord("q"):
            print("\n⛔ Thoát!")
            break

        elif key == ord("c"):
            sentence.clear()
            predictions.clear()
            sequence.clear()
            current_res = None
            print("🗑️ Đã xóa câu")

        elif key == ord("s"):
            result_str = "_".join(sentence) if sentence else "empty"
            cv2.imwrite(f"./output_{result_str}.jpg", frame)
            print(f"💾 Đã lưu ảnh: output_{result_str}.jpg")

cap.release()
cv2.destroyAllWindows()
print("✅ Kết thúc!")