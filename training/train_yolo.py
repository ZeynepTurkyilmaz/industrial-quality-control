from ultralytics import YOLO
import os

# Fix for OMP: Error #15
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import shutil

def train():
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model

    # Train the model
    # Assuming script is run from project root
    print("Starting training...")
    try:
        results = model.train(
            data='training/data.yaml',
            epochs=80,
            imgsz=640,
            batch=16,
            project='runs',
            name='detect',
            exist_ok=True,  # overwrite existing experiment named 'detect'
        )
    except Exception as e:
        print(f"Error during training: {e}")
        return

    # Path to best weights
    # Ultralytics saves to project/name/weights/best.pt
    best_weights = 'runs/detect/weights/best.pt'
    
    # Target path
    target_dir = 'api/model'
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, 'best.pt')
    
    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_path)
        print(f"✅ Success! Best model saved to {target_path}")
    else:
        print(f"❌ Error: Best weights not found at {best_weights}")

if __name__ == '__main__':
    train()
