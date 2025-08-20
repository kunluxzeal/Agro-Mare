from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
import json
import uvicorn
import socket
import os
from fastapi.middleware.cors import CORSMiddleware
from utils import analyze_image_with_openai, SYSTEM_PROMPT, VECTOR_STORE_ID

import requests


MODEL_PATH = Path("../../training/hierarchical_models_mobilenetv2")
DISEASE_CONFIDENCE_THRESHOLD = 0.4
PLANT_TYPE_CONFIDENCE_THRESHOLD = 0.6

# Mode can be "offline" or "online"
MODE = os.getenv("MODE", "offline").lower()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HierarchicalModel:
    def __init__(self):
        self.plant_classifier = None
        self.disease_classifiers = {}
        self.plant_type_classes = []
        self.disease_classes = {}

    def load_models(self, model_path):
        try:
            self.plant_classifier = tf.keras.models.load_model(
                model_path / 'plant_type_classifier.h5'
            )
            with open(model_path / 'class_mappings.json') as f:
                mappings = json.load(f)
            self.plant_type_classes = mappings['plant_types']
            self.disease_classes = mappings['disease_classes']
            for plant_type in self.plant_type_classes:
                model_file = model_path / f'{plant_type}_disease_classifier.h5'
                if model_file.exists():
                    self.disease_classifiers[plant_type] = {
                        'model': tf.keras.models.load_model(model_file),
                        'class_names': self.disease_classes[plant_type]
                    }
            print("Models loaded successfully")
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

model_loader = HierarchicalModel()
model_loader.load_models(MODEL_PATH)

def preprocess_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


def internet_connected(timeout=3):
    """Check if we can reach OpenAI API."""
    try:
        requests.head("https://api.openai.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API", "mode": MODE}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        
        # Automatically determine mode
        if internet_connected():
            mode = "online"
        else:
            mode = "offline"

        # OFFLINE MODE
        if mode == "offline":
            image_array = preprocess_image(image_data)
            image_batch = np.expand_dims(image_array, axis=0)

            plant_predictions = model_loader.plant_classifier.predict(image_batch)
            plant_type_idx = np.argmax(plant_predictions[0])
            plant_confidence = plant_predictions[0][plant_type_idx]

            result = {
                "mode": "offline",
                "plant_type": "unknown",
                "plant_confidence": float(plant_confidence),
                "disease": "unknown",
                "disease_confidence": 0.0,
                "is_healthy": False,
                "message": "Offline mode - no AI advice",
                "treatment": None
            }

            if plant_confidence >= PLANT_TYPE_CONFIDENCE_THRESHOLD:
                predicted_plant = model_loader.plant_type_classes[plant_type_idx]
                result["plant_type"] = predicted_plant

                if predicted_plant in model_loader.disease_classifiers:
                    disease_model = model_loader.disease_classifiers[predicted_plant]['model']
                    disease_classes = model_loader.disease_classifiers[predicted_plant]['class_names']
                    disease_predictions = disease_model.predict(image_batch)
                    disease_idx = np.argmax(disease_predictions[0])
                    disease_confidence = disease_predictions[0][disease_idx]

                    if disease_confidence >= DISEASE_CONFIDENCE_THRESHOLD:
                        predicted_disease = disease_classes[disease_idx]
                        result.update({
                            "disease": predicted_disease,
                            "disease_confidence": float(disease_confidence),
                            "is_healthy": "healthy" in predicted_disease.lower(),
                            "message": f"Disease detected: {predicted_disease} (offline mode)"
                        })
            return result

        # ONLINE MODE
        elif mode == "online":
            if not internet_connected():
                raise HTTPException(status_code=503, detail="No internet connection for online mode")

            temp_image_path = "temp_upload.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_data)

            treatment_json = analyze_image_with_openai(
                image_path=temp_image_path,
                result_text="Analyze and return disease prediction + treatment advice in JSON format"
            )

            # Check if voice output is requested
            accept_header = request.headers.get("accept", "")
            if "audio/" in accept_header and treatment_json.get('voice_output'):
                return StreamingResponse(
                    BytesIO(treatment_json['voice_output']),
                    media_type="audio/mpeg",
                    headers={"X-Analysis": json.dumps(treatment_json)}
                )
            
            # Remove voice_output from JSON response as it's binary data
            if 'voice_output' in treatment_json:
                del treatment_json['voice_output']
                
            return {
                "mode": "online",
                "analysis": treatment_json
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid MODE setting. Use 'offline' or 'online'.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
