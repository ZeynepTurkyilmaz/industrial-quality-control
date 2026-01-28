import cv2
import numpy as np
import os
import glob
import shutil

# Paths
IMG_DIR = "data/yolo/images/train"
LABEL_DIR = "data/yolo/labels/train"
TARGET_CLASS = 0 # Scratch

def load_yolo_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                # cls, x, y, w, h
                boxes.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return boxes

def save_yolo_label(path, boxes):
    with open(path, 'w') as f:
        for b in boxes:
            f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")

def augment_image(img_path, label_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    boxes = load_yolo_label(label_path)
    
    # Only augment if it contains the target class
    has_target = any(b[0] == TARGET_CLASS for b in boxes)
    if not has_target:
        return

    basename = os.path.splitext(os.path.basename(img_path))[0]
    print(f"Augmenting {basename}...")

    # 1. Flip Horizontal
    img_flip_h = cv2.flip(img, 1) # 1 = horizontal
    boxes_flip_h = []
    for b in boxes:
        # x_new = 1 - x_old
        boxes_flip_h.append([b[0], 1.0 - b[1], b[2], b[3], b[4]])
    
    save_aug(basename + "_flip_h", img_flip_h, boxes_flip_h)

    # 2. Flip Vertical
    img_flip_v = cv2.flip(img, 0) # 0 = vertical
    boxes_flip_v = []
    for b in boxes:
        # y_new = 1 - y_old
        boxes_flip_v.append([b[0], b[1], 1.0 - b[2], b[3], b[4]])
    
    save_aug(basename + "_flip_v", img_flip_v, boxes_flip_v)

    # 3. Rotate 180 (Flip H + V)
    img_rot180 = cv2.flip(img, -1) # -1 = both
    boxes_rot180 = []
    for b in boxes:
        # x_new = 1 - x_old, y_new = 1 - y_old
        boxes_rot180.append([b[0], 1.0 - b[1], 1.0 - b[2], b[3], b[4]])
    
    save_aug(basename + "_rot180", img_rot180, boxes_rot180)

    # 4. Brightness Increase
    # Convert to HSV, increase V
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)
    img_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    save_aug(basename + "_bright", img_bright, boxes) # Boxes don't change

    # 5. Brightness Decrease
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 0.8, 0, 255)
    img_dark = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    save_aug(basename + "_dark", img_dark, boxes) # Boxes don't change


def save_aug(name, img, boxes):
    cv2.imwrite(os.path.join(IMG_DIR, name + ".png"), img)
    save_yolo_label(os.path.join(LABEL_DIR, name + ".txt"), boxes)

def main():
    img_files = glob.glob(os.path.join(IMG_DIR, "*.png"))
    for f in img_files:
        label_file = os.path.join(LABEL_DIR, os.path.basename(f).replace(".png", ".txt"))
        augment_image(f, label_file)

if __name__ == "__main__":
    main()
