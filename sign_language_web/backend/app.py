from pathlib import Path
import tempfile

import cv2
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config import (
    MODEL_DIR,
    MODEL_FILENAME,
    NORM_FILENAME,
    LABELS_FILENAME,
    DISPLAY_LABELS_FILENAME,
    SEQUENCE_LENGTH,
    THRESHOLD,
    MOTION_THRESHOLD,
    IDLE_RESET_AFTER,
    PREDICTION_WINDOW,
    MAJORITY_COUNT,
)
from services.model_service import ModelService
from services.mediapipe_service import MediaPipeService
from services.recognition_service import RecognitionService
from utils.image_utils import decode_base64_image, resize_keep_ratio

ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
CORS(app)

model_service = None
mediapipe_service = None
recognition_service = None


def init_services():
    global model_service, mediapipe_service, recognition_service
    model_service = ModelService(MODEL_DIR, MODEL_FILENAME, NORM_FILENAME, LABELS_FILENAME, DISPLAY_LABELS_FILENAME)
    mediapipe_service = MediaPipeService()
    recognition_service = RecognitionService(
        model_service=model_service,
        sequence_length=SEQUENCE_LENGTH,
        motion_threshold=MOTION_THRESHOLD,
        idle_reset_after=IDLE_RESET_AFTER,
        threshold=THRESHOLD,
        prediction_window=PREDICTION_WINDOW,
        majority_count=MAJORITY_COUNT,
    )


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_service is not None,
        "labels": model_service.labels if model_service else [],
        "display_labels": model_service.display_labels if model_service else {},
        "sequence_length": SEQUENCE_LENGTH,
        "threshold": THRESHOLD,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    return jsonify(recognition_service.reset())


@app.route("/api/clear", methods=["POST"])
def clear():
    return jsonify(recognition_service.clear_sentence())


@app.route("/api/backspace", methods=["POST"])
def backspace():
    return jsonify(recognition_service.remove_last_word())


@app.route("/api/predict-frame", methods=["POST"])
def predict_frame():
    payload = request.get_json(silent=True) or {}
    image = payload.get("image")
    if not image:
        return jsonify({"status": "error", "message": "Thiếu trường image"}), 400

    try:
        frame = decode_base64_image(image)
        frame = resize_keep_ratio(frame, max_width=640)
        keypoints, meta = mediapipe_service.process_frame(frame)
        result = recognition_service.process_keypoints(keypoints, meta)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/predict-video", methods=["POST"])
def predict_video():
    """Upload video dự phòng cho demo. Backend đọc frame và chạy cùng pipeline."""
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "Thiếu file video"}), 400

    video_file = request.files["video"]
    recognition_service.reset()

    suffix = Path(video_file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    cap = cv2.VideoCapture(video_path)
    results = []
    frame_idx = 0
    # Lấy mỗi 2 frame để nhẹ hơn, có thể đổi sample_step = 1 nếu cần
    sample_step = int(request.form.get("sample_step", 2))
    sample_step = max(1, sample_step)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_step != 0:
                continue

            frame = resize_keep_ratio(frame, max_width=640)
            keypoints, meta = mediapipe_service.process_frame(frame)
            result = recognition_service.process_keypoints(keypoints, meta)
            if result.get("added_word"):
                results.append({
                    "frame": frame_idx,
                    "word": result["added_word"],
                    "word_display": result.get("added_word_display"),
                    "sentence": result["sentence"],
                    "sentence_display": result.get("sentence_display", []),
                })
    finally:
        cap.release()
        Path(video_path).unlink(missing_ok=True)

    return jsonify({
        "status": "ok",
        "results": results,
        "sentence": recognition_service.sentence,
        "sentence_display": recognition_service._sentence_display(),
        "sentence_text": " ".join(recognition_service._sentence_display()),
        "frames_processed": frame_idx,
    })


if __name__ == "__main__":
    init_services()
    app.run(host="127.0.0.1", port=5000, debug=False)
