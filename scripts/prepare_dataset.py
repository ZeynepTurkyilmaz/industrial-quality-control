import cv2
import os
import random
import shutil

#for path
RAW_DATA_DIR = "data/raw/metal_nut"
YOLO_DATA_DIR = "data/yolo"

IMAGE_OUT_DIR = os.path.join(YOLO_DATA_DIR, "images")
LABEL_OUT_DIR = os.path.join(YOLO_DATA_DIR, "labels")

#classes
CLASSES = {
    "scratch": 0,
    "bent": 1
}

SPLIT_RATIO = 0.8  # %80 train / %20 val


def ensure_dirs():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(IMAGE_OUT_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABEL_OUT_DIR, split), exist_ok=True)

def mask_to_bboxes(mask):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 10]

def process_defect(defect_name):
    img_dir = os.path.join(RAW_DATA_DIR, "test", defect_name)
    mask_dir = os.path.join(RAW_DATA_DIR, "ground_truth", defect_name)

    images = sorted(os.listdir(img_dir))
    random.shuffle(images)

    split_idx = int(len(images) * SPLIT_RATIO)
    split_map = {
        "train": images[:split_idx],
        "val": images[split_idx:]
    }

    for split, img_list in split_map.items():
        for img_name in img_list:
            img_path = os.path.join(img_dir, img_name)
            mask_name = img_name.replace(".png", "_mask.png")
            mask_path = os.path.join(mask_dir, mask_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            bboxes = mask_to_bboxes(mask)
            if not bboxes:
                continue

            label_lines = []
            for x, y, bw, bh in bboxes:
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h

                label_lines.append(
                    f"{CLASSES[defect_name]} "
                    f"{x_center:.6f} {y_center:.6f} "
                    f"{bw_norm:.6f} {bh_norm:.6f}"
                )

            # Image kopyala
            out_img_path = os.path.join(
                IMAGE_OUT_DIR, split, img_name
            )
            shutil.copy(img_path, out_img_path)

            # Label yaz
            label_path = os.path.join(
                LABEL_OUT_DIR, split, img_name.replace(".png", ".txt")
            )
            with open(label_path, "w") as f:
                f.write("\n".join(label_lines))


if __name__ == "__main__":
    ensure_dirs()
    for defect in CLASSES.keys():
        process_defect(defect)

    print("✅ YOLO dataset başarıyla oluşturuldu.")
