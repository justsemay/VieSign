# src/word/evaluate.py
import numpy as np
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from config.word_config import ACTIONS, MODEL_DIR

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Chọn model cần đánh giá
# ============================================================
MODEL_NAME = "cnn_bilstm_word_standard_24052026_03"

MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
NORM_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_norm_stats.npz")
TEST_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_test_data.npz")
CONFIG_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_config.npz")

REPORT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_classification_report.txt")
METRICS_JSON_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_metrics.json")
CM_COUNT_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_confusion_matrix_count.png")
CM_NORM_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_confusion_matrix_normalized.png")


# ============================================================
# 2. Kiểm tra file
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy model: {MODEL_PATH}")

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy test data: {TEST_PATH}")

print("=" * 70)
print("📊 MODEL EVALUATION")
print("=" * 70)
print(f"📦 Model     : {MODEL_PATH}")
print(f"📊 Test data : {TEST_PATH}")

if os.path.exists(CONFIG_PATH):
    config = np.load(CONFIG_PATH, allow_pickle=True)
    print(f"⚙️  Config    : {CONFIG_PATH}")

    if "use_shoulder_normalize" in config:
        print(f"   - use_shoulder_normalize: {config['use_shoulder_normalize']}")
    if "sequence_length" in config:
        print(f"   - sequence_length       : {config['sequence_length']}")
    if "num_classes" in config:
        print(f"   - num_classes           : {config['num_classes']}")
else:
    print("⚠️ Không tìm thấy config file. Bỏ qua phần config.")


# ============================================================
# 3. Load model
# ============================================================
model = load_model(MODEL_PATH)
print(f"\n✅ Model loaded")
print(f"   Input shape : {model.input_shape}")
print(f"   Output shape: {model.output_shape}")


# ============================================================
# 4. Load test data
# ============================================================
test_data = np.load(TEST_PATH)

if "X_test" not in test_data or "y_test" not in test_data:
    raise KeyError("❌ test_data.npz phải có X_test và y_test")

X_test = test_data["X_test"].astype("float32")
y_test = test_data["y_test"]

print(f"\n📐 X_test shape: {X_test.shape}")
print(f"🏷️  y_test shape: {y_test.shape}")

if X_test.shape[0] != y_test.shape[0]:
    raise ValueError("❌ Số lượng X_test và y_test không khớp")

if len(ACTIONS) != model.output_shape[-1]:
    print("⚠️ Cảnh báo: Số ACTIONS khác số output của model")
    print(f"   len(ACTIONS)       = {len(ACTIONS)}")
    print(f"   model output class = {model.output_shape[-1]}")


# ============================================================
# 5. Xử lý y_true
# ============================================================
if len(y_test.shape) == 2:
    y_true = np.argmax(y_test, axis=1)
else:
    y_true = y_test.astype(int)

unique_labels, counts = np.unique(y_true, return_counts=True)
print("\n📌 Test label distribution:")
for label, count in zip(unique_labels, counts):
    action_name = ACTIONS[label] if label < len(ACTIONS) else f"class_{label}"
    print(f"   {action_name:15s}: {count}")


# ============================================================
# 6. Normalize lại hay không?
# ============================================================
# Trong train.py hiện tại, bạn lưu X_test SAU khi đã normalize.
# Vì vậy mặc định KHÔNG normalize lại.
#
# Nếu sau này bạn đổi train.py để lưu X_test trước normalize,
# thì đổi USE_NORM = True.
USE_NORM = False

if USE_NORM:
    if not os.path.exists(NORM_PATH):
        raise FileNotFoundError(f"❌ Không tìm thấy norm stats: {NORM_PATH}")

    norm = np.load(NORM_PATH)
    mean = norm["mean"]
    std = norm["std"]

    X_test = (X_test - mean) / std
    print(f"\n✅ Đã normalize lại bằng: {NORM_PATH}")
else:
    print("\nℹ️ Không normalize lại vì X_test đã được lưu sau normalize.")


# ============================================================
# 7. Predict
# ============================================================
print("\n🚀 Đang predict trên test set...")
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

confidence = np.max(y_prob, axis=1)
avg_confidence = float(np.mean(confidence))
min_confidence = float(np.min(confidence))
max_confidence = float(np.max(confidence))

print(f"✅ Predict xong")
print(f"   Avg confidence: {avg_confidence:.4f}")
print(f"   Min confidence: {min_confidence:.4f}")
print(f"   Max confidence: {max_confidence:.4f}")


# ============================================================
# 8. Overall metrics
# ============================================================
acc = accuracy_score(y_true, y_pred)

precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n📈 Overall Metrics:")
print(f"Accuracy           : {acc:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")


# ============================================================
# 9. Classification report
# ============================================================
print("\n📋 Classification Report:")

report = classification_report(
    y_true,
    y_pred,
    target_names=ACTIONS,
    zero_division=0
)

print(report)


# ============================================================
# 10. Confusion Matrix
# ============================================================
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(ACTIONS))))

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=ACTIONS,
    yticklabels=ACTIONS
)

plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CM_COUNT_PATH, dpi=300)
plt.show()

print(f"💾 Đã lưu confusion matrix count: {CM_COUNT_PATH}")


cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=ACTIONS,
    yticklabels=ACTIONS
)

plt.title(f"Normalized Confusion Matrix - {MODEL_NAME}")
plt.ylabel("Thực tế")
plt.xlabel("Dự đoán")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CM_NORM_PATH, dpi=300)
plt.show()

print(f"💾 Đã lưu confusion matrix normalized: {CM_NORM_PATH}")


# ============================================================
# 11. Top-3 Accuracy
# ============================================================
top3 = np.argsort(y_prob, axis=1)[:, -3:]

top3_correct = np.mean([
    y_true[i] in top3[i]
    for i in range(len(y_true))
])

print(f"\n🎯 Top-3 Accuracy: {top3_correct:.4f}")


# ============================================================
# 12. In danh sách mẫu dự đoán sai
# ============================================================
wrong_indices = np.where(y_true != y_pred)[0]

print("\n❌ Wrong Predictions:")
if len(wrong_indices) == 0:
    print("   Không có mẫu dự đoán sai.")
else:
    for idx in wrong_indices:
        true_name = ACTIONS[y_true[idx]]
        pred_name = ACTIONS[y_pred[idx]]
        conf = confidence[idx]

        top3_names = [
            f"{ACTIONS[i]}({y_prob[idx][i] * 100:.1f}%)"
            for i in top3[idx][::-1]
        ]

        print(
            f"   #{idx:04d} | True: {true_name:15s} | "
            f"Pred: {pred_name:15s} | Conf: {conf:.4f} | "
            f"Top3: {', '.join(top3_names)}"
        )


# ============================================================
# 13. Save report txt
# ============================================================
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write("=" * 70 + "\n\n")

    f.write("Overall Metrics\n")
    f.write("=" * 70 + "\n")
    f.write(f"Accuracy           : {acc:.4f}\n")
    f.write(f"Macro Precision    : {precision_macro:.4f}\n")
    f.write(f"Macro Recall       : {recall_macro:.4f}\n")
    f.write(f"Macro F1-score     : {f1_macro:.4f}\n")
    f.write(f"Weighted Precision : {precision_weighted:.4f}\n")
    f.write(f"Weighted Recall    : {recall_weighted:.4f}\n")
    f.write(f"Weighted F1-score  : {f1_weighted:.4f}\n")
    f.write(f"Top-3 Accuracy     : {top3_correct:.4f}\n")
    f.write(f"Avg Confidence     : {avg_confidence:.4f}\n")
    f.write(f"Min Confidence     : {min_confidence:.4f}\n")
    f.write(f"Max Confidence     : {max_confidence:.4f}\n\n")

    f.write("Classification Report\n")
    f.write("=" * 70 + "\n")
    f.write(report)

    f.write("\n\nWrong Predictions\n")
    f.write("=" * 70 + "\n")
    if len(wrong_indices) == 0:
        f.write("Không có mẫu dự đoán sai.\n")
    else:
        for idx in wrong_indices:
            true_name = ACTIONS[y_true[idx]]
            pred_name = ACTIONS[y_pred[idx]]
            conf = confidence[idx]
            top3_names = [
                f"{ACTIONS[i]}({y_prob[idx][i] * 100:.1f}%)"
                for i in top3[idx][::-1]
            ]
            f.write(
                f"#{idx:04d} | True: {true_name:15s} | "
                f"Pred: {pred_name:15s} | Conf: {conf:.4f} | "
                f"Top3: {', '.join(top3_names)}\n"
            )

print(f"💾 Đã lưu report: {REPORT_PATH}")


# ============================================================
# 14. Save metrics json
# ============================================================
metrics = {
    "model_name": MODEL_NAME,
    "accuracy": float(acc),
    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),
    "top3_accuracy": float(top3_correct),
    "avg_confidence": avg_confidence,
    "min_confidence": min_confidence,
    "max_confidence": max_confidence,
    "num_test_samples": int(len(y_true)),
    "num_wrong": int(len(wrong_indices))
}

with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)

print(f"💾 Đã lưu metrics json: {METRICS_JSON_PATH}")

print("\n✅ Đánh giá hoàn tất!")