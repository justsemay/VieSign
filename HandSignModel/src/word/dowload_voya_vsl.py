from datasets import load_dataset
import numpy as np
import os

# ── Cấu hình ────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "VOYA_VSL")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Tải dataset từ HuggingFace ───────────────────────────────
print("⏬ Đang tải dataset Kateht/VOYA_VSL...")
dataset = load_dataset("Kateht/VOYA_VSL")
print(dataset)

# ── Split train / val / test ─────────────────────────────────
split_1 = dataset["train"].train_test_split(test_size=0.2, seed=42)
split_2 = split_1["train"].train_test_split(test_size=0.1, seed=42)

train_ds = split_2["train"]
val_ds   = split_2["test"]
test_ds  = split_1["test"]

print(f"\n📊 Kích thước:")
print(f"  Train : {len(train_ds)} samples")
print(f"  Val   : {len(val_ds)} samples")
print(f"  Test  : {len(test_ds)} samples")

# ── Chuyển sang numpy và lưu ─────────────────────────────────
def save_split(ds, name):
    X = np.array(ds["sequences"])   # (N, 60, 1605)
    y = np.array(ds["labels"])      # (N,)
    np.save(os.path.join(SAVE_DIR, f"X_{name}.npy"), X)
    np.save(os.path.join(SAVE_DIR, f"y_{name}.npy"), y)
    print(f"  ✅ Lưu {name}: X{X.shape}  y{y.shape}")

print("\n💾 Đang lưu file .npy...")
save_split(train_ds, "train")
save_split(val_ds,   "val")
save_split(test_ds,  "test")

print(f"\n✅ Xong! Dữ liệu lưu tại: {SAVE_DIR}")