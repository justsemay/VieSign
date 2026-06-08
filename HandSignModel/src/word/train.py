import os
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Conv1D,
    LSTM,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
# ── Import config ───────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import ACTIONS, DATA_PATH, MODEL_DIR, SEQUENCE_LENGTH


# ============================================================
# 1. Experiment Config
# ============================================================
VOCAB_TYPE = "word"
MODEL_TYPE = "bilstm_aug"
NORM_TYPE = "standard"

USE_SHOULDER_NORMALIZE = False
USE_AUGMENTATION = True
AUG_PER_SAMPLE = 1

EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
LABEL_SMOOTHING = 0.1

TEST_SIZE = 0.2
VAL_SIZE_FROM_TEMP = 0.5
RANDOM_STATE = 42


# ============================================================
# 2. Model Naming
# ============================================================
def create_model_paths():
    date_str = datetime.now().strftime("%d%m%Y")
    prefix = f"{MODEL_TYPE}_{VOCAB_TYPE}_{NORM_TYPE}_{date_str}_"

    os.makedirs(MODEL_DIR, exist_ok=True)

    existing = [
        f for f in os.listdir(MODEL_DIR)
        if f.startswith(prefix) and f.endswith(".keras")
    ]

    run_number = len(existing) + 1
    model_name = f"{prefix}{run_number:02d}"

    paths = {
        "model_name": model_name,
        "model": os.path.join(MODEL_DIR, f"{model_name}.keras"),
        "norm": os.path.join(MODEL_DIR, f"{model_name}_norm_stats.npz"),
        "test": os.path.join(MODEL_DIR, f"{model_name}_test_data.npz"),
        "config": os.path.join(MODEL_DIR, f"{model_name}_config.npz"),
        "history_acc": os.path.join(MODEL_DIR, f"{model_name}_accuracy.png"),
        "history_loss": os.path.join(MODEL_DIR, f"{model_name}_loss.png"),
        "history_npz": os.path.join(MODEL_DIR, f"{model_name}_history.npz"),
    }

    print(f"🏷️  Model name : {model_name}")
    print(f"💾 Save path  : {paths['model']}")
    print(f"📦 Run #{run_number} hôm nay ({date_str})\n")

    return paths


# ============================================================
# 3. Data Loading
# ============================================================
def load_dataset():
    print("📂 Đang load data...")

    X, y = [], []

    for label_idx, action in enumerate(ACTIONS):
        action_dir = os.path.join(DATA_PATH, action)

        if not os.path.exists(action_dir):
            print(f"  ⚠️  Không tìm thấy: {action_dir}")
            continue

        seqs = sorted(
            [
                d for d in os.listdir(action_dir)
                if os.path.isdir(os.path.join(action_dir, d))
            ],
            key=lambda x: int(x)
        )

        valid_count = 0

        for seq in seqs:
            frames = []
            valid = True

            for frame_num in range(SEQUENCE_LENGTH):
                npy_path = os.path.join(action_dir, seq, f"{frame_num}.npy")

                if not os.path.exists(npy_path):
                    valid = False
                    break

                frames.append(np.load(npy_path))

            if valid and len(frames) == SEQUENCE_LENGTH:
                X.append(frames)
                y.append(label_idx)
                valid_count += 1

        print(f"  ✅ {action:15s}: {valid_count}/{len(seqs)} valid sequences")

    X = np.array(X, dtype="float32")
    y = np.array(y)

    print(f"\n📊 Dataset: {X.shape}  |  Classes: {len(ACTIONS)}")

    for i, action in enumerate(ACTIONS):
        count = np.sum(y == i)
        mark = "⚠️ " if count < 30 else "✅"
        print(f"  {mark} {action:15s}: {count} samples")

    return X, y


# ============================================================
# 4. Optional Shoulder Normalize
# ============================================================
def normalize_by_shoulders(sequence):
    """
    sequence shape: (T, 225)
    225 = pose(33*3) + left_hand(21*3) + right_hand(21*3)
    """
    sequence = sequence.copy()
    seq_3d = sequence.reshape(sequence.shape[0], -1, 3)

    left_shoulder = seq_3d[:, 11, :]
    right_shoulder = seq_3d[:, 12, :]

    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.linalg.norm(left_shoulder - right_shoulder, axis=1, keepdims=True)

    valid = scale.squeeze() > 1e-4

    seq_norm = seq_3d.copy()
    seq_norm[valid] = seq_norm[valid] - center[valid, None, :]
    seq_norm[valid] = seq_norm[valid] / (scale[valid, None, :] + 1e-6)

    return seq_norm.reshape(sequence.shape)


def apply_optional_shoulder_normalize(X):
    if not USE_SHOULDER_NORMALIZE:
        print("ℹ️  Không dùng Shoulder-relative normalization.")
        return X

    X_norm = np.array(
        [normalize_by_shoulders(seq) for seq in X],
        dtype="float32"
    )

    print("✅ Đã áp dụng Shoulder-relative + Scale normalization")
    return X_norm

def add_noise(seq, noise_std=0.002):
    """
    Thêm nhiễu nhẹ vào landmark.
    Giúp model chịu được sai số nhỏ từ MediaPipe.
    """
    noise = np.random.normal(0, noise_std, seq.shape)
    return seq + noise


def shift_xy(seq, max_shift=0.015):
    """
    Dịch nhẹ toàn bộ landmark theo trục x/y.
    Giả lập người đứng lệch trái/phải hoặc cao/thấp trong khung hình.
    """
    seq = seq.copy()
    seq_3d = seq.reshape(seq.shape[0], -1, 3)

    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)

    seq_3d[:, :, 0] += shift_x
    seq_3d[:, :, 1] += shift_y

    return seq_3d.reshape(seq.shape)


def scale_xy(seq, scale_range=(0.96, 1.04)):
    """
    Scale nhẹ tọa độ x/y quanh tâm.
    Giả lập người đứng gần/xa camera.
    """
    seq = seq.copy()
    seq_3d = seq.reshape(seq.shape[0], -1, 3)

    scale = np.random.uniform(scale_range[0], scale_range[1])
    center = np.mean(seq_3d[:, :, :2], axis=(0, 1), keepdims=True)

    seq_3d[:, :, :2] = (seq_3d[:, :, :2] - center) * scale + center

    return seq_3d.reshape(seq.shape)


def time_shift(seq, max_shift=2):
    """
    Dịch chuỗi frame sớm/muộn.
    Giả lập lúc bắt đầu thu bị sớm hoặc muộn vài frame.
    """
    shift = np.random.randint(-max_shift, max_shift + 1)

    if shift == 0:
        return seq.copy()

    if shift > 0:
        pad = np.repeat(seq[0:1], shift, axis=0)
        return np.concatenate([pad, seq[:-shift]], axis=0)

    shift = abs(shift)
    pad = np.repeat(seq[-1:], shift, axis=0)
    return np.concatenate([seq[shift:], pad], axis=0)


def frame_dropout(seq, drop_prob=0.02):
    """
    Giả lập MediaPipe bị mất tracking nhẹ vài frame.
    Frame bị dropout sẽ được thay bằng frame trước đó.
    """
    seq = seq.copy()

    for i in range(1, seq.shape[0]):
        if np.random.rand() < drop_prob:
            seq[i] = seq[i - 1]

    return seq


def augment_sequence(seq):
    """
    Tạo một bản augmented từ sequence landmark.
    seq shape: (SEQUENCE_LENGTH, feature_dim)
    """
    aug = seq.copy().astype("float32")

    if np.random.rand() < 0.7:
        aug = add_noise(aug, noise_std=0.002)

    if np.random.rand() < 0.4:
        aug = shift_xy(aug, max_shift=0.015)

    if np.random.rand() < 0.4:
        aug = scale_xy(aug, scale_range=(0.96, 1.04))

    if np.random.rand() < 0.5:
        aug = time_shift(aug, max_shift=2)

    if np.random.rand() < 0.3:
        aug = frame_dropout(aug, drop_prob=0.02)

    return aug


def augment_training_data(X_train, y_train, label_train):
    """
    Chỉ augment tập train.
    Không augment val/test để tránh đánh giá ảo.
    """
    if not USE_AUGMENTATION:
        print("ℹ️  Không dùng data augmentation.")
        return X_train, y_train, label_train

    X_new = []
    y_new = []
    label_new = []

    for seq, y_onehot, label in zip(X_train, y_train, label_train):
        # Giữ bản gốc
        X_new.append(seq)
        y_new.append(y_onehot)
        label_new.append(label)

        # Tạo thêm bản augmented
        for _ in range(AUG_PER_SAMPLE):
            X_new.append(augment_sequence(seq))
            y_new.append(y_onehot)
            label_new.append(label)

    X_new = np.array(X_new, dtype="float32")
    y_new = np.array(y_new)
    label_new = np.array(label_new)

    print(
        f"✅ Đã augmentation train: "
        f"{len(X_train)} → {len(X_new)} samples "
        f"(AUG_PER_SAMPLE={AUG_PER_SAMPLE})"
    )

    return X_new, y_new, label_new
# ============================================================
# 5. Train / Val / Test Split
# ============================================================
def split_dataset(X, y):
    Y = to_categorical(y, num_classes=len(ACTIONS))

    X_train, X_temp, y_train, y_temp, label_train, label_temp = train_test_split(
        X,
        Y,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE_FROM_TEMP,
        stratify=label_temp,
        random_state=RANDOM_STATE
    )

    print(f"\n🔀 Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, label_train


# ============================================================
# 6. Standardization
# ============================================================
def standardize_data(X_train, X_val, X_test):
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")
    X_test = X_test.astype("float32")

    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


# ============================================================
# 7. Save Training Artifacts
# ============================================================
def save_training_artifacts(paths, mean, std, X_test, y_test):
    np.savez(
        paths["norm"],
        mean=mean,
        std=std
    )

    np.savez(
        paths["test"],
        X_test=X_test,
        y_test=y_test
    )

    np.savez(
        paths["config"],
        use_shoulder_normalize=USE_SHOULDER_NORMALIZE,
        use_augmentation=USE_AUGMENTATION,
        aug_per_sample=AUG_PER_SAMPLE,
        model_type=MODEL_TYPE,
        norm_type=NORM_TYPE,
        sequence_length=SEQUENCE_LENGTH,
        num_classes=len(ACTIONS)
    )

    print(f"💾 Norm stats đã lưu: {paths['norm']}")
    print(f"💾 Test data đã lưu: {paths['test']}")
    print(f"💾 Config đã lưu: {paths['config']}")


# ============================================================
# 8. Class Weights
# ============================================================
def get_class_weights(y):
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    )

    class_weight_dict = dict(enumerate(class_weights))

    print(
        f"⚖️  Class weights: "
        f"{ {ACTIONS[k]: round(v, 2) for k, v in class_weight_dict.items()} }"
    )

    return class_weight_dict


# ============================================================
# 9. Build Model
# ============================================================
def build_cnn_bilstm_model(feature_dim, model_name):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, feature_dim)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(0.3),

        Conv1D(
            filters=128,
            kernel_size=3,
            padding="same",
            activation="relu"
        ),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),

        Bidirectional(LSTM(64)),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),

        Dense(len(ACTIONS), activation="softmax")
    ], name=model_name)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=["accuracy"]
    )

    return model

def build_bilstm_model(feature_dim, model_name):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, feature_dim)),

        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(
            LSTM(
                128,
                return_sequences=True,
                recurrent_dropout=0.2
            )
        ),
        Dropout(0.4),

        Bidirectional(
            LSTM(
                64,
                recurrent_dropout=0.2
            )
        ),
        Dropout(0.4),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),

        Dense(len(ACTIONS), activation="softmax")
    ], name=model_name)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=["accuracy"]
    )

    return model
# ============================================================
# 10. Callbacks
# ============================================================
def get_callbacks(model_path):
    return [
        ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        )
    ]


# ============================================================
# 11. Evaluation
# ============================================================
def evaluate_model(model, X_test, y_test):
    print("\n📈 Đánh giá trên test set:")

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss: {loss:.4f}  |  Accuracy: {acc * 100:.2f}%")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=ACTIONS))

# ============================================================
# 12. Plot Training History
# ============================================================
def plot_training_history(history, paths):
    """
    Vẽ và lưu biểu đồ accuracy/loss trong quá trình huấn luyện.
    Dùng cho báo cáo thực nghiệm.
    """
    hist = history.history

    # Lưu history dạng npz để có thể mở lại sau này nếu cần
    np.savez(
        paths["history_npz"],
        accuracy=np.array(hist.get("accuracy", [])),
        val_accuracy=np.array(hist.get("val_accuracy", [])),
        loss=np.array(hist.get("loss", [])),
        val_loss=np.array(hist.get("val_loss", []))
    )

    # -----------------------------
    # Accuracy chart
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(hist["accuracy"], label="Train Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths["history_acc"], dpi=300)
    plt.close()

    # -----------------------------
    # Loss chart
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(hist["loss"], label="Train Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths["history_loss"], dpi=300)
    plt.close()

    print(f"📊 Accuracy chart đã lưu: {paths['history_acc']}")
    print(f"📉 Loss chart đã lưu: {paths['history_loss']}")
    print(f"💾 History data đã lưu: {paths['history_npz']}")
# ============================================================
# 13. Main
# ============================================================
def main():
    paths = create_model_paths()

    X, y = load_dataset()

    X = apply_optional_shoulder_normalize(X)

    X_train, X_val, X_test, y_train, y_val, y_test, label_train = split_dataset(X, y)

    # X_train, y_train, label_train = augment_training_data(
    #     X_train,
    #     y_train,
    #     label_train
    # )

    X_train, X_val, X_test, mean, std = standardize_data(
        X_train,
        X_val,
        X_test
    )

    save_training_artifacts(
        paths=paths,
        mean=mean,
        std=std,
        X_test=X_test,
        y_test=y_test
    )

    class_weight_dict = get_class_weights(label_train)

    model = build_bilstm_model(
        feature_dim=X_train.shape[2],
        model_name=paths["model_name"]
    )

    model.summary()

    callbacks = get_callbacks(paths["model"])

    print("\n🚀 Bắt đầu train...\n")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    plot_training_history(history, paths)
    evaluate_model(model, X_test, y_test)

    print(f"\n✅ Model đã lưu: {paths['model']}")


if __name__ == "__main__":
    main()