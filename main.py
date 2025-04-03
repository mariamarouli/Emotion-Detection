import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import os

# Suppress TensorFlow info and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # '0' = all logs, '1' = warnings, '2' = errors, '3' = nothing


import tensorflow as tf
import uvicorn
from io import BytesIO
from PIL import Image
from typing import Tuple


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.keras.models.load_model('model.h5')

class_names = ['angry', 'disgusted', 'fearful', 
            'happy', 'neutral', 'sad', 'surprised']



def read_file_as_image(data) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('L')
    img_resized = img.resize((48, 48))
    image = np.array(img_resized) / 255.0  
    
    return image, img_resized.size


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image, image_size = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        'class': predicted_class,
        'confidence': confidence
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    with open('main.html', 'r') as f:
        return f.read()


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)