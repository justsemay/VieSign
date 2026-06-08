# src/word/test_tflite_inference.py
import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.word_config import MODEL_DIR


MODEL_NAME = "cnn_bilstm_word_standard_24052026_03"

ASSET_DIR = os.path.join(MODEL_DIR, "android_assets", MODEL_NAME)

TFLITE_PATH = os.path.join(ASSET_DIR, "model.tflite")
LABELS_PATH = os.path.join(ASSET_DIR, "labels.txt")
TEST_DATA_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_test_data.npz")


if not os.path.exists(TFLITE_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy TFLite model: {TFLITE_PATH}")

if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy test data: {TEST_DATA_PATH}")


# Load labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines() if line.strip()]


# Load test data
test_data = np.load(TEST_DATA_PATH)

X_test = test_data["X_test"].astype("float32")
y_test = test_data["y_test"]

if len(y_test.shape) == 2:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test.astype(int)


# Load TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 70)
print("🧪 TEST TFLITE INFERENCE")
print("=" * 70)
print("Input details:")
print(input_details)
print("\nOutput details:")
print(output_details)


# Test thử 20 mẫu đầu
correct = 0
num_test = len(X_test)

for i in range(num_test):
    input_data = np.expand_dims(X_test[i], axis=0).astype("float32")

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(output))
    true_idx = int(y_true[i])

    if pred_idx == true_idx:
        correct += 1

accuracy = correct / num_test
print(f"\n🎯 TFLite Accuracy on full test set: {accuracy:.4f} ({correct}/{num_test})")