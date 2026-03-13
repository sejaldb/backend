from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import cv2
import random

app = FastAPI(title="Deepfake Detection API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------
# Response Models
# -------------------------------
class FrameRecord(BaseModel):
    frame_index: int
    fake_probability: float

class UploadResponse(BaseModel):
    deepfakeProbability: float
    realProbability: float
    confidence: str
    verdict: str
    suspiciousFrames: List[FrameRecord]
    framesAnalyzed: int
    message: str

# -------------------------------
# Dummy CNN model (replace with real model)
# -------------------------------
def dummy_cnn_predict(frame: np.ndarray):
    """
    Simulate a CNN deepfake prediction.
    Returns fake probability between 0 and 1.
    """
    fake_prob = random.uniform(0, 1)
    return {"fake_probability": round(fake_prob, 4), "real_probability": round(1-fake_prob, 4)}

# -------------------------------
# Score aggregation
# -------------------------------
def aggregate_scores(frames: List[FrameRecord]):
    if not frames:
        return {
            "deepfakeProbability": 0.0,
            "realProbability": 1.0,
            "confidence": "low",
            "verdict": "uncertain",
            "suspiciousFrames": [],
        }
    fake_probs = np.array([f.fake_probability for f in frames])
    avg_fake = np.mean(fake_probs)
    avg_real = 1 - avg_fake

    # Confidence based on number of frames and extremeness
    extremeness = abs(avg_fake - 0.5) * 2
    if len(frames) < 3 or extremeness < 0.3:
        confidence = "low"
    elif len(frames) < 10 or extremeness < 0.6:
        confidence = "medium"
    else:
        confidence = "high"

    verdict = "deepfake" if avg_fake > 0.6 else "real" if avg_fake < 0.4 else "uncertain"
    suspicious_frames = [f for f in frames if f.fake_probability > 0.65]

    return {
        "deepfakeProbability": float(avg_fake),
        "realProbability": float(avg_real),
        "confidence": confidence,
        "verdict": verdict,
        "suspiciousFrames": suspicious_frames,
    }

# -------------------------------
# Helper: Extract frames from video
# -------------------------------
def extract_frames(video_bytes: bytes, frame_step: int = 30):
    frames = []
    video_array = np.frombuffer(video_bytes, np.uint8)
    cap = cv2.VideoCapture(io.BytesIO(video_bytes))
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_step == 0:
            frames.append(frame)
        count += 1
    cap.release()
    return frames

# -------------------------------
# POST /upload
# -------------------------------
@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    filename = file.filename
    content = await file.read()

    frames: List[FrameRecord] = []

    if file.content_type.startswith("image/"):
        # Single frame for image
        image = np.array(Image.open(io.BytesIO(content)).convert("RGB"))
        prediction = dummy_cnn_predict(image)
        frames.append(FrameRecord(frame_index=0, fake_probability=prediction["fake_probability"]))
    elif file.content_type.startswith("video/"):
        # Video: extract frames
        video_frames = extract_frames(content, frame_step=30)  # every 30 frames
        for idx, frame in enumerate(video_frames):
            prediction = dummy_cnn_predict(frame)
            frames.append(FrameRecord(frame_index=idx, fake_probability=prediction["fake_probability"]))
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    aggregated = aggregate_scores(frames)
    return UploadResponse(
        framesAnalyzed=len(frames),
        message=f"Analyzed {len(frames)} frame(s) from {filename}",
        **aggregated
    )

# -------------------------------
# Health check
# -------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}