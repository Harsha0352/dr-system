# Diabetic Retinopathy Detection System

This is an AI-powered system designed to detect and grade Diabetic Retinopathy (DR) from retinal fundus images. It aims to enable fast and accessible screening, potentially preventing vision loss through early detection.

## Technology Stack

### Backend
- **Framework**: FastAPI (Python)
- **Machine Learning**: TensorFlow / Keras (ResNet50)
- **Deployment**: Render

### Frontend
- **Framework**: React.js (Vite)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Deployment**: Vercel

## Project Structure

```
/
├── backend/            # FastAPI backend application
│   ├── main.py         # API entry point
│   ├── model.py        # Model definition and training logic
│   ├── utils.py        # Image preprocessing utilities
│   ├── dr_model.h5     # Trained model weights
│   └── requirements.txt # Python dependencies
└── frontend/           # React frontend application
    ├── src/            # Source code
    ├── package.json    # Node dependencies
    └── tailwind.config.js # Tailwind configuration
```

## Setup & Installation

### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`.

## API Usage

### Health Check
- **Endpoint**: `GET /`
- **Response**: `{"message": "Welcome to DR Detection API"}`

### Predict Image
- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: multipart/form-data`
- **Body**: A form-data `file` field containing the retinal image.
- **Response**:
  ```json
  {
    "filename": "image.jpg",
    "prediction_class": 0,
    "prediction_label": "No DR",
    "confidence": 0.98
  }
  ```

## Deployment

### Backend (Render)
1. Push this repository to GitHub.
2. Create a new **Web Service** on Render.
3. Connect your repository.
4. Settings:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**: Add `PYTHON_VERSION` = `3.9.0` (or your local version).

### Frontend (Vercel)
1. Import the repository into Vercel.
2. Settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Environment Variables**: Add `VITE_API_URL` set to your **Render Backend URL** (e.g., `https://your-backend.onrender.com`).
3. Deploy.

## Deliverables Checklist
- [x] FastAPI backend with image upload & prediction endpoints.
- [x] React frontend for image upload and result display.
- [ ] Deployed backend URL (Render).
- [ ] Deployed frontend URL (Vercel).
- [x] Short README explaining setup and API usage.
