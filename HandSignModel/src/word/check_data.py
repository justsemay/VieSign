import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import ACTIONS, DATA_PATH, NUM_SEQUENCES, SEQUENCE_LENGTH

print("🔍 Kiểm tra data...\n")
missing = []

for action in ACTIONS:
    for seq in range(NUM_SEQUENCES):
        for frame in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, action, str(seq), f"{frame}.npy")
            if not os.path.exists(path):
                missing.append(path)

if missing:
    print(f"❌ Thiếu {len(missing)} file:")
    for m in missing[:10]:
        print(f"   {m}")
else:
    print("✅ Data đầy đủ!")
    print(f"   {len(ACTIONS)} từ × {NUM_SEQUENCES} videos × {SEQUENCE_LENGTH} frames")
    print(f"   = {len(ACTIONS) * NUM_SEQUENCES * SEQUENCE_LENGTH} file .npy")

    # Kiểm tra shape
    sample = np.load(os.path.join(DATA_PATH, ACTIONS[0], "0", "0.npy"))
    print(f"\n📐 Shape mỗi frame: {sample.shape}")
    print(f"   → Pose: 33×3=99 | Left Hand: 21×3=63 | Right Hand: 21×3=63 | Tổng: {sample.shape[0]}")