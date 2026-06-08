import cv2
import numpy as np
import mediapipe as mp
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import ACTIONS, DATA_PATH, NUM_SEQUENCES, SEQUENCE_LENGTH

# ── MediaPipe setup ──────────────────────────────────────────
mp_holistic  = mp.solutions.holistic
mp_draw      = mp.solutions.drawing_utils

def extract_keypoints(results):
    """Trích xuất 225 keypoints từ MediaPipe Holistic"""
    pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 3)

    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # (225,)

def draw_landmarks(frame, results):
    """Vẽ các điểm landmark lên frame"""
    # Pose
    mp_draw.draw_landmarks(frame, results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_draw.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_draw.DrawingSpec(color=(80,256,121), thickness=1))
    # Tay trái
    mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(121,44,250), thickness=2))
    # Tay phải
    mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=3),
        mp_draw.DrawingSpec(color=(245,66,230), thickness=2))

# ── Tạo thư mục ─────────────────────────────────────────────
for action in ACTIONS:
    for seq in range(NUM_SEQUENCES):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)
print(f"✅ Đã tạo thư mục cho {len(ACTIONS)} từ")

# ── Thu thập data ────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

with mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as holistic:

    for action_idx, action in enumerate(ACTIONS):
        print(f"\n📌 Từ {action_idx+1}/{len(ACTIONS)}: [{action}]")

        for seq in range(NUM_SEQUENCES):
            last_frame_path = os.path.join(DATA_PATH, action, str(seq), f"{SEQUENCE_LENGTH-1}.npy")
            if os.path.exists(last_frame_path):
                print(f"  ⏭️  Bo qua seq {seq+1} (da co data)")
                continue
            
            
            # ── Màn hình chờ ────────────────────────────────
            while True:
                ret, frame = cap.read()
                if not ret: continue

                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (frame.shape[1], 80), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                cv2.putText(frame,
                    f"Tu: [{action}]  |  Video: {seq+1}/{NUM_SEQUENCES}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.putText(frame,
                    "Nhan SPACE de bat dau  |  Q de thoat",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 1)

                cv2.imshow("Thu thap data VSL", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == ord(' '): break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("⛔ Đã thoát!")
                    sys.exit()

            # ── Đếm ngược 3 giây ────────────────────────────
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                cv2.putText(frame, str(countdown),
                    (frame.shape[1]//2 - 30, frame.shape[0]//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 8)
                cv2.imshow("Thu thap data VSL", frame)
                cv2.waitKey(1000)

            # ── Quay 30 frame ────────────────────────────────
            for frame_num in range(SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret: continue

                # Xử lý MediaPipe
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Vẽ landmarks
                draw_landmarks(frame, results)

                # Extract & lưu keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(seq), str(frame_num))
                np.save(npy_path, keypoints)

                # Progress bar
                progress = int((frame_num + 1) / SEQUENCE_LENGTH * 400)
                cv2.rectangle(frame,
                    (20, frame.shape[0]-50),
                    (420, frame.shape[0]-25), (50,50,50), -1)
                cv2.rectangle(frame,
                    (20, frame.shape[0]-50),
                    (20 + progress, frame.shape[0]-25), (0,255,0), -1)
                cv2.putText(frame,
                    f"{action} | Video {seq+1}/{NUM_SEQUENCES} | Frame {frame_num+1}/{SEQUENCE_LENGTH}",
                    (20, frame.shape[0]-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                cv2.imshow("Thu thap data VSL", frame)
                cv2.waitKey(30)

            print(f"  ✅ Xong video {seq+1}/{NUM_SEQUENCES}")

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Thu thập data hoàn tất!")
print(f"   Tổng: {len(ACTIONS)} từ × {NUM_SEQUENCES} video × {SEQUENCE_LENGTH} frame")