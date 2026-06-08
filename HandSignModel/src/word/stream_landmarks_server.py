import asyncio
import json
import time

import cv2
import mediapipe as mp
import numpy as np
import websockets


mp_holistic = mp.solutions.holistic


def extract_keypoints(results):
    pose = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 3)
    )

    left_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )

    right_hand = (
        np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )

    features = np.concatenate([pose, left_hand, right_hand]).astype(np.float32)

    if features.shape[0] != 225:
        raise ValueError(f"Feature size must be 225, got {features.shape[0]}")

    return features


async def stream_landmarks(websocket):
    print("✅ Android connected")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    frame_count = 0
    last_log_time = time.time()

    with mp_holistic.Holistic(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as holistic:
        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print("❌ Cannot read frame")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = holistic.process(image)

                features = extract_keypoints(results)

                payload = {
                    "timestamp": int(time.time() * 1000),
                    "features": features.tolist(),
                    "hasPose": results.pose_landmarks is not None,
                    "hasLeftHand": results.left_hand_landmarks is not None,
                    "hasRightHand": results.right_hand_landmarks is not None,
                }

                await websocket.send(json.dumps(payload))

                frame_count += 1
                now = time.time()

                if now - last_log_time >= 1.0:
                    print(
                        f"FPS={frame_count} | "
                        f"pose={payload['hasPose']} | "
                        f"left={payload['hasLeftHand']} | "
                        f"right={payload['hasRightHand']}"
                    )
                    frame_count = 0
                    last_log_time = now

                await asyncio.sleep(1 / 15)

        except websockets.ConnectionClosed:
            print("⚠️ Android disconnected")

        finally:
            cap.release()
            print("✅ Camera released")


async def main():
    print("🚀 WebSocket server running at ws://0.0.0.0:8765")
    print("📱 Android emulator should connect to ws://10.0.2.2:8765")

    async with websockets.serve(stream_landmarks, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())