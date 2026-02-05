# Diabetic Retinopathy Detection System

This project is an AI-powered system for detecting and grading Diabetic Retinopathy (DR) from retinal fundus images. It consists of a FastAPI backend and a React + Tailwind CSS frontend.

## Directory Structure

- `backend/`: FastAPI application, CNN model, and training scripts.
- `frontend/`: React + TypeScript application.

## Prerequisites

- **Python 3.8+**
- **Node.js 16+** (Required for frontend)

## Dataset

The system is configured to use the dataset located at:
`d:/DRDS_HARSHA/dr_unified_v2/dr_unified_v2`

## Setup & Running

### 1. Backend (Python/FastAPI)

Navigate to the backend directory:
```bash
cd backend
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the server:
```bash
uvicorn main:app --reload
```
The backend API will run at `http://localhost:8000`.

*Note: On first run, if no trained model is found, it will load an untuned ResNet50 for demo purposes. You can run the training logic in `model.py` if you wish to train on the real dataset.*

### 2. Frontend (React)

Navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies (ensure Node.js is installed):
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
The frontend will run at `http://localhost:5173`.

## Usage

1. Open `http://localhost:5173` in your browser.
2. Drag and drop a retinal image (or select one).
3. The system will analyze the image and display the DR grade (0-4) and confidence score.
