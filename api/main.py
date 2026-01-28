from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os
import shutil
from database.db import save_defect, Base, engine

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Industrial Quality Control API")

# Model path
MODEL_PATH = "api/model/best.pt"
model = None

@app.on_event("startup")
async def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Model not found at {MODEL_PATH}. Inference will fail.")

@app.post("/detect")
async def detect_defect(file: UploadFile = File(...)):
    global model
    if model is None:
        # Try to load again if it wasn't there at startup
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            raise HTTPException(status_code=503, detail="Model not loaded")

    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Inference
    results = model(img)
    
    # Process results
    detections = []
    
    # Assuming single object or taking the highest confidence one?
    # Requirement: "tespit edilen kusur sınıfı ile güven skorlarını JSON olarak döndür"
    # We'll return all detections
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            detection = {
                "defect_type": class_name,
                "confidence": conf,
                "bbox": box.xywh.tolist()[0] # x,y,w,h
            }
            detections.append(detection)
            
            # Save to DB - Log each detection?
            # Requirement: "her başarılı tespitte bu tabloya kayıt atan bir fonksiyon ekle"
            save_defect(defect_type=class_name, confidence=conf)

    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
