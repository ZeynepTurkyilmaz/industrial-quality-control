from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "api/model/best.pt"
IMG_2 = "data/yolo/images/val/002.png"
IMG_15 = "data/yolo/images/val/015.png"

def test_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    # Standard confidence test
    print("\n--- Testing with Standard Confidence (0.25) ---")
    for img_path in [IMG_2, IMG_15]:
        if not os.path.exists(img_path):
            continue
            
        print(f"\nImage: {img_path}")
        results = model(img_path)
        for r in results:
            print(f"Detections: {len(r.boxes)}")
            for box in r.boxes:
                print(f"Class: {int(box.cls)} ({model.names[int(box.cls)]}), Conf: {float(box.conf):.4f}, Box: {box.xywh.tolist()}")

if __name__ == "__main__":
    test_model()
