from ultralytics import YOLO
import sys
import os

# Paths
MODEL_PATH = "api/model/best.pt"
IMAGE_PATH = "data/yolo/images/val/002.png"

def run_debug():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Running inference on {IMAGE_PATH}...")
    results = model(IMAGE_PATH, conf=0.01)

    print("\n--- Results ---")
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects.")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            print(f"- Class: {name} (ID: {cls_id}), Confidence: {conf:.4f}")

if __name__ == "__main__":
    run_debug()
