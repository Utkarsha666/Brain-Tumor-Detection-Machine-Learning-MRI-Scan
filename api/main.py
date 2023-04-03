from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from skimage import transform
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
__class_name_to_number = {}
class_number_to_name = {}

MODEL = tf.keras.models.load_model("../models/2")
input_shape = (224, 224)
@app.get("/ping")
async def ping():
    return "Hello I am Alive"


def read_file_as_image(data):
    image = (Image.open(BytesIO(data)))
    return image


with open('class_dictionary.json', "r") as f:
    __class_name_to_number = json.load(f)
    class_number_to_name = {v:k for k, v in __class_name_to_number.items()}

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    np_image = np.array(image).astype('float32')/255
    np_image = transform.resize(np_image, (224, 224, 3))
    img_batch = np.expand_dims(np_image, 0) #since our training datas were in batches so our Neural Network expets tensor size (Batch_size, Width, Height, Channels)

    # Now predict our image
    predictions = MODEL.predict(img_batch)

    predicted_class = np.argmax(predictions[0])

    predicted_name = class_number_to_name[predicted_class]
    confidence = np.max(predictions[0])

    return {
        'class' : predicted_name,
        'confidence' : float(confidence)
    }

if __name__=="__main__":
    uvicorn.run(app, host='localhost', port=8000)