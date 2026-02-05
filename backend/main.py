from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
# Defer heavy imports to prevent startup timeout/crash
# from model import load_trained_model, NUM_CLASSES 
# from utils import read_image_file, preprocess_image

app = FastAPI(title="Diabetic Retinopathy Detection API")

# Global State
model = None
CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# CORS setup
origins = [
    "http://localhost:5173",
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

@app.on_event("startup")
async def startup_event():
    global model
    print("🚀 App starting... Lazy loading model.")
    try:
        from model import load_trained_model
        model = load_trained_model()
        if model:
            print("✅ Model loaded successfully!")
        else:
            print("⚠️ Model failed to load (or Mock Mode active).")
    except Exception as e:
        print(f"❌ Critical error loading model module: {e}")
        model = None

@app.get("/")
def read_root():
    return {"message": "Welcome to DR Detection API"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # NUCLEAR OPTION: Return hardcoded success immediately
    # This proves if the deployment is actually updating or stuck on old code.
    return {
        "filename": getattr(file, "filename", "demo.jpg"),
        "prediction_class": 0,
        "prediction_label": "No DR (Safe Mode)",
        "confidence": 0.99
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
