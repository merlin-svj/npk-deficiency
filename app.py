from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os

app = FastAPI()

# Load your trained model
model = YOLO("best.pt")

# Serve static HTML (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    # Save uploaded image
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    result = model(img)
    probs = result[0].probs.data.cpu().numpy()  # Get probabilities as numpy array
    class_names = result[0].names

    predictions = []
    for idx, prob in enumerate(probs):
        if prob > 0.5:
            predictions.append(class_names[idx])

    # Always include the top prediction if nothing else passes threshold
    if not predictions:
        top_class = result[0].probs.top1
        predictions.append(class_names[top_class])

    return {"prediction": ", ".join(predictions)}
