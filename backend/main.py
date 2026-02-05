from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uvicorn
from model import load_trained_model, NUM_CLASSES
from utils import read_image_file, preprocess_image
import tensorflow as tf

app = FastAPI(title="Diabetic Retinopathy Detection API")

# CORS setup
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:3000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = load_trained_model()

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

@app.get("/")
def read_root():
    return {"message": "Welcome to DR Detection API"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file:
        return {"error": "No file uploaded"}
    
    try:
        # Read and preprocess
        image_data = await file.read()
        image = read_image_file(image_data)
        processed_image = preprocess_image(image)
        
        # Predict
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            "filename": file.filename,
            "prediction_class": int(predicted_class),
            "prediction_label": CLASS_NAMES.get(int(predicted_class), "Unknown"),
            "confidence": confidence
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
