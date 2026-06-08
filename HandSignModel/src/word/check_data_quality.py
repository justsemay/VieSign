# src/word/check_data_quality.py
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config.word_config import ACTIONS, DATA_PATH, SEQUENCE_LENGTH

# ── Ngưỡng đánh giá ──────────────────────────────────────────
MIN_HAND_RATIO  = 0.70   # Tối thiểu 70% frame phải có tay
MIN_POSE_RATIO  = 0.90   # Tối thiểu 90% frame phải có pose
MIN_MOTION      = 0.002  # Motion quá thấp → video đứng im (nghi ngờ)
MAX_MOTION      = 0.15   # Motion quá cao  → video bị nhiễu

def analyze_sequence(seq_dir):
    """Phân tích 1 video, trả về dict thống kê"""
    stats = {
        "hand_frames":  0,   # Frame có ít nhất 1 tay
        "pose_frames":  0,   # Frame có pose
        "lhand_frames": 0,   # Frame có tay trái
        "rhand_frames": 0,   # Frame có tay phải
        "missing_files": 0,
        "motion_values": [],
        "total": SEQUENCE_LENGTH
    }

    prev_kp = None
    for f in range(SEQUENCE_LENGTH):
        npy_path = os.path.join(seq_dir, f"{f}.npy")
        if not os.path.exists(npy_path):
            stats["missing_files"] += 1
            continue

        kp   = np.load(npy_path)
        pose = kp[:99]
        lh   = kp[99:162]
        rh   = kp[162:225]

        if np.any(pose != 0): stats["pose_frames"]  += 1
        if np.any(lh   != 0): stats["lhand_frames"] += 1
        if np.any(rh   != 0): stats["rhand_frames"] += 1
        if np.any(lh   != 0) or np.any(rh != 0):
            stats["hand_frames"] += 1

        # Motion giữa frame liền kề
        if prev_kp is not None:
            motion = np.mean(np.abs(kp[99:] - prev_kp[99:]))
            stats["motion_values"].append(motion)
        prev_kp = kp

    return stats

def grade_sequence(stats):
    """Chấm điểm video: GOOD / WARN / BAD"""
    total       = stats["total"]
    hand_ratio  = stats["hand_frames"]  / total
    pose_ratio  = stats["pose_frames"]  / total
    avg_motion  = np.mean(stats["motion_values"]) if stats["motion_values"] else 0
    missing     = stats["missing_files"]

    issues = []

    if missing > 0:
        issues.append(f"thiếu {missing} file")
    if hand_ratio < MIN_HAND_RATIO:
        issues.append(f"tay chỉ detect {hand_ratio:.0%}")
    if pose_ratio < MIN_POSE_RATIO:
        issues.append(f"pose chỉ detect {pose_ratio:.0%}")
    if avg_motion < MIN_MOTION:
        issues.append(f"ít chuyển động ({avg_motion:.4f}) → nghi đứng im")
    if avg_motion > MAX_MOTION:
        issues.append(f"quá nhiễu ({avg_motion:.4f})")

    if not issues:
        return "GOOD", issues
    elif len(issues) == 1:
        return "WARN", issues
    else:
        return "BAD", issues

# ── Main ─────────────────────────────────────────────────────
print("=" * 65)
print("📊 KIỂM TRA CHẤT LƯỢNG DATA")
print("=" * 65)

summary = {
    "total_videos": 0,
    "good": 0,
    "warn": 0,
    "bad":  0,
    "bad_list": []   # [(action, seq, issues)]
}

for action in ACTIONS:
    if action == "IDLE":
        continue

    action_dir = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_dir):
        print(f"\n⚠️  KHÔNG TÌM THẤY: {action}")
        continue

    seqs = sorted(
        [d for d in os.listdir(action_dir)
         if os.path.isdir(os.path.join(action_dir, d)) and d.isdigit()],
        key=lambda x: int(x)
    )

    good_count = warn_count = bad_count = 0
    bad_seqs   = []

    for seq in seqs:
        seq_dir = os.path.join(action_dir, seq)
        stats   = analyze_sequence(seq_dir)
        grade, issues = grade_sequence(stats)

        summary["total_videos"] += 1

        if grade == "GOOD":
            good_count += 1
            summary["good"] += 1
        elif grade == "WARN":
            warn_count += 1
            summary["warn"] += 1
            bad_seqs.append((seq, grade, issues))
        else:
            bad_count += 1
            summary["bad"] += 1
            bad_seqs.append((seq, grade, issues))
            summary["bad_list"].append((action, seq, issues))

    # In kết quả từng action
    total_seq  = len(seqs)
    icon = "✅" if bad_count == 0 and warn_count == 0 else \
           "⚠️ " if bad_count == 0 else "❌"

    print(f"\n{icon} [{action:15s}]  "
          f"Total:{total_seq:3d}  "
          f"✅{good_count:3d}  ⚠️{warn_count:2d}  ❌{bad_count:2d}")

    for seq, grade, issues in bad_seqs:
        icon2 = "⚠️ " if grade == "WARN" else "❌"
        print(f"     {icon2} Video #{seq:>3s}: {' | '.join(issues)}")

# ── Tổng kết ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("📋 TỔNG KẾT")
print("=" * 65)
print(f"  Tổng video  : {summary['total_videos']}")
print(f"  ✅ GOOD     : {summary['good']}  ({summary['good']/max(summary['total_videos'],1)*100:.1f}%)")
print(f"  ⚠️  WARN     : {summary['warn']}  ({summary['warn']/max(summary['total_videos'],1)*100:.1f}%)")
print(f"  ❌ BAD      : {summary['bad']}  ({summary['bad']/max(summary['total_videos'],1)*100:.1f}%)")

if summary["bad_list"]:
    print(f"\n🗑️  Danh sách cần xóa/quay lại:")
    for action, seq, issues in summary["bad_list"]:
        print(f"   {action}/video_{seq}: {' | '.join(issues)}")

    # Hỏi có muốn xóa tự động không
    print("\n" + "-" * 65)
    ans = input("❓ Tự động XÓA các video BAD? (y/n): ").strip().lower()
    if ans == 'y':
        deleted = 0
        for action, seq, _ in summary["bad_list"]:
            seq_dir = os.path.join(DATA_PATH, action, seq)
            import shutil
            shutil.rmtree(seq_dir)
            print(f"   🗑️  Đã xóa: {action}/video_{seq}")
            deleted += 1
        print(f"\n✅ Đã xóa {deleted} video kém chất lượng")
        print("💡 Nhớ chạy lại collect_data.py để thu bù!")
    else:
        print("ℹ️  Không xóa. Xem lại thủ công rồi quyết định.")

print("\n✅ Kiểm tra hoàn tất!")