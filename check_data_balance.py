import os
from collections import Counter
import glob

TRAIN_LABELS_PATH = "data/yolo/labels/train"
VAL_LABELS_PATH = "data/yolo/labels/val"

def check_labels(path, split_name):
    print(f"\n--- Checking {split_name} set ---")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return

    label_files = glob.glob(os.path.join(path, "*.txt"))
    print(f"Found {len(label_files)} label files.")

    class_counts = Counter()
    files_with_scratch = []
    
    for lf in label_files:
        with open(lf, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_id = int(parts[0])
                    class_counts[cls_id] += 1
                    if cls_id == 0:
                        files_with_scratch.append(os.path.basename(lf))

    print("Class Counts:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} instances")

    print(f"Files containing Class 0 (scratch): {len(files_with_scratch)}")
    if len(files_with_scratch) > 0:
        print(f"  First 5 files with scratch: {files_with_scratch[:5]}")
    else:
        print("  WARNING: No scratch labels found!")

if __name__ == "__main__":
    check_labels(TRAIN_LABELS_PATH, "Train")
    check_labels(VAL_LABELS_PATH, "Validation")
