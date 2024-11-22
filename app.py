from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf
import numpy as np
import io
from mangum import Mangum  # Add Mangum for Vercel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "model.tflite"  # Use relative path for model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = [
    "burger", "butter_naan", "chai", "chapati", "chole_bhature", "dal_makhani",
    "dhokla", "fried_rice", "idli", "jalebi", "kaathi_rolls", "kadai_paneer",
    "kulfi", "masala_dosa", "momos", "paani_puri", "pakode", "pav_bhaji",
    "pizza", "samosa"
]

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading image: {e}")

    try:
        input_data = preprocess_image(image, (input_details[0]['shape'][1], input_details[0]['shape'][2]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {e}")

    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index']).flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during model inference: {e}")

    try:
        predicted_class_index = np.argmax(output_data)
        predicted_probability = output_data[predicted_class_index]

        if predicted_probability < 0.5:
            return {"prediction": "NONE"}

        predicted_class = class_names[predicted_class_index]
        return {
            "prediction": predicted_class,
            "confidence": f"{predicted_probability * 100:.2f}%"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction result: {e}")

@app.get("/")
async def root():
    return {"message": "Welcome to the Food Prediction API!"}

handler = Mangum(app)  # Use Mangum to handle the request
