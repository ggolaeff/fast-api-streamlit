from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import time

# Конфигурация
INPUT_SIZE = (150, 150)
CLASS_NAMES = ['Lilly', 'Lotus', 'Orchid', 'Sunflower', 'Tulip']

app = FastAPI(
    title="Flower Classification API",
    version="1.0.0",
    description="API для классификации изображений цветов"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('basic_cnn_model.h5')

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize(INPUT_SIZE)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "Требуется изображение")
    
    try:
        start_time = time.time()
        img_bytes = await file.read()
        img_processed = preprocess_image(img_bytes)
        preds = model.predict(img_processed)[0]
        
        return {
            "class": CLASS_NAMES[np.argmax(preds)],
            "confidence": float(np.max(preds)),
            "probabilities": {name: float(preds[i]) for i, name in enumerate(CLASS_NAMES)},
            "processing_time": f"{time.time() - start_time:.3f} сек"
        }
    except Exception as e:
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")

@app.get("/")
async def info():
    return {
        "model": "Basic CNN",
        "input_size": INPUT_SIZE,
        "available_classes": CLASS_NAMES
    }
