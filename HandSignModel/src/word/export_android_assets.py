# src/word/export_android_assets.py
import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.word_config import ACTIONS, MODEL_DIR, SEQUENCE_LENGTH


# ============================================================
# Chọn model muốn deploy
# ============================================================
MODEL_NAME = "cnn_bilstm_word_standard_24052026_03"
# Ví dụ:
# MODEL_NAME = "cnn_bilstm_word_standard_24052026_02"
# MODEL_NAME = "cnn_bilstm_aug_word_standard_25052026_01"

KERAS_MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
NORM_STATS_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_norm_stats.npz")
CONFIG_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_config.npz")

EXPORT_DIR = os.path.join(MODEL_DIR, "android_assets", MODEL_NAME)
os.makedirs(EXPORT_DIR, exist_ok=True)

TFLITE_PATH = os.path.join(EXPORT_DIR, "model.tflite")
LABELS_PATH = os.path.join(EXPORT_DIR, "labels.txt")
MEAN_PATH = os.path.join(EXPORT_DIR, "norm_mean.json")
STD_PATH = os.path.join(EXPORT_DIR, "norm_std.json")
ANDROID_CONFIG_PATH = os.path.join(EXPORT_DIR, "model_config.json")


# ============================================================
# Check files
# ============================================================
if not os.path.exists(KERAS_MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy model: {KERAS_MODEL_PATH}")

if not os.path.exists(NORM_STATS_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy norm stats: {NORM_STATS_PATH}")


print("=" * 70)
print("🚀 EXPORT ANDROID ASSETS")
print("=" * 70)
print(f"📦 Keras model : {KERAS_MODEL_PATH}")
print(f"📊 Norm stats  : {NORM_STATS_PATH}")
print(f"📁 Export dir  : {EXPORT_DIR}")


# ============================================================
# Load Keras model
# ============================================================
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

print("\n✅ Loaded Keras model")
print(f"Input shape : {model.input_shape}")
print(f"Output shape: {model.output_shape}")


# ============================================================
# Convert Keras -> SavedModel -> TFLite
# ============================================================
print("\n🔄 Exporting Keras model to SavedModel...")

SAVED_MODEL_DIR = os.path.join(EXPORT_DIR, "saved_model_tmp")

# Nếu folder cũ tồn tại thì xóa trước
import shutil
if os.path.exists(SAVED_MODEL_DIR):
    shutil.rmtree(SAVED_MODEL_DIR)

# Với Keras 3, dùng model.export()
model.export(SAVED_MODEL_DIR)

print(f"✅ SavedModel exported: {SAVED_MODEL_DIR}")


print("\n🔄 Converting SavedModel to TFLite with SELECT_TF_OPS...")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("✅ Converted TFLite successfully with SELECT_TF_OPS.")
print(f"💾 Saved TFLite: {TFLITE_PATH}")

# ============================================================
# Export labels.txt
# ============================================================
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    for action in ACTIONS:
        f.write(action + "\n")

print(f"💾 Saved labels: {LABELS_PATH}")


# ============================================================
# Export mean/std
# ============================================================
norm = np.load(NORM_STATS_PATH)

mean = norm["mean"].astype("float32")
std = norm["std"].astype("float32")

mean_flat = mean.reshape(-1).tolist()
std_flat = std.reshape(-1).tolist()

with open(MEAN_PATH, "w", encoding="utf-8") as f:
    json.dump(mean_flat, f)

with open(STD_PATH, "w", encoding="utf-8") as f:
    json.dump(std_flat, f)

print(f"💾 Saved mean: {MEAN_PATH}")
print(f"💾 Saved std : {STD_PATH}")


# ============================================================
# Export model_config.json
# ============================================================
use_shoulder_normalize = False
norm_type = "standard"
model_type = "cnn_bilstm"

if os.path.exists(CONFIG_PATH):
    cfg = np.load(CONFIG_PATH, allow_pickle=True)

    if "use_shoulder_normalize" in cfg:
        use_shoulder_normalize = bool(cfg["use_shoulder_normalize"])

    if "norm_type" in cfg:
        norm_type = str(cfg["norm_type"])

    if "model_type" in cfg:
        model_type = str(cfg["model_type"])


model_config = {
    "model_name": MODEL_NAME,
    "model_type": model_type,
    "norm_type": norm_type,
    "sequence_length": SEQUENCE_LENGTH,
    "feature_dim": 225,
    "num_classes": len(ACTIONS),
    "use_shoulder_normalize": use_shoulder_normalize,
    "input_shape": [1, SEQUENCE_LENGTH, 225],
    "output_shape": [1, len(ACTIONS)],
    "requires_select_tf_ops": True
}

with open(ANDROID_CONFIG_PATH, "w", encoding="utf-8") as f:
    json.dump(model_config, f, ensure_ascii=False, indent=4)

print(f"💾 Saved config: {ANDROID_CONFIG_PATH}")

print("\n✅ Export hoàn tất!")
print(f"📁 Copy folder này vào Android assets:")
print(EXPORT_DIR)