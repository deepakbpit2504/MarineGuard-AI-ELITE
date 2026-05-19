from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from model import predict
from logger import log_event

app = FastAPI(title="MarineGuard AI")

@app.get("/")
def home():
    return {"status": "MarineGuard AI running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = predict(img)

    log_event(results)

    return {
        "status": "success",
        "detections": results
    }
