import os
import io
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import numpy as np

app = FastAPI(title="Model Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = os.environ.get("MODEL_PATH", "model/mobilenetv2_cat_skin_disease.h5")
PREPROCESS_MODE = os.environ.get("PREPROCESS_MODE", "0-1")  # '0-1' or '-1-1'


def load_model(path: str):
    import tensorflow as tf

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    model = tf.keras.models.load_model(path, compile=False)
    return model


def get_target_size(model):
    try:
        shape = model.input_shape
    except Exception:
        try:
            shape = model.inputs[0].shape
        except Exception:
            shape = None
    if not shape:
        return (224, 224)
    try:
        h = int(shape[1]) if shape[1] is not None else 224
        w = int(shape[2]) if shape[2] is not None else 224
        return (h, w)
    except Exception:
        return (224, 224)


def preprocess_image_bytes(data: bytes, target_size, mode: str = "0-1"):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype(np.float32)
    if mode == "0-1":
        arr = arr / 255.0
    elif mode == "-1-1":
        arr = (arr / 255.0 - 0.5) * 2.0
    arr = np.expand_dims(arr, 0)
    return arr


class PredictResponse(BaseModel):
    predictions: List[float]


@app.on_event("startup")
def startup_event():
    global MODEL
    try:
        MODEL = load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    except Exception as e:
        MODEL = None
        print(f"Model not loaded at startup: {e}")


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    try:
        target = get_target_size(MODEL)
        arr = preprocess_image_bytes(data, target, PREPROCESS_MODE)
        preds = MODEL.predict(arr)
        preds = np.asarray(preds)
        if preds.ndim == 2:
            probs = preds[0].tolist()
        else:
            probs = preds.flatten().tolist()
        return {"predictions": probs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
