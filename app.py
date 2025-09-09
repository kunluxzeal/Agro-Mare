# from fastapi import FastAPI, File, UploadFile, HTTPException, Request
# from fastapi.responses import StreamingResponse, FileResponse
# import uuid
# from io import BytesIO
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import os
# import tensorflow as tf
# from pathlib import Path
# import json
# import uvicorn
# import socket
# import os
# from fastapi.middleware.cors import CORSMiddleware
# from utils import analyze_image_with_openai, SYSTEM_PROMPT, VECTOR_STORE_ID
# from fastapi.staticfiles import StaticFiles
# import base64
# import requests

# MODEL_PATH = Path("hierarchical_models_mobilenetv2")
# DISEASE_CONFIDENCE_THRESHOLD = 0.4
# PLANT_TYPE_CONFIDENCE_THRESHOLD = 0.6

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Mode can be "offline" or "online"
# MODE = os.getenv("MODE", "offline").lower()


# app = FastAPI()

# app.mount("/static", StaticFiles(directory="static"), name="static")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class HierarchicalModel:
#     def __init__(self):
#         self.plant_classifier = None
#         self.disease_classifiers = {}
#         self.plant_type_classes = []
#         self.disease_classes = {}

#     def load_models(self, model_path):
#         try:
#             self.plant_classifier = tf.keras.models.load_model(
#                 model_path / 'plant_type_classifier.h5',
#                 compile=False
#             )
#             with open(model_path / 'class_mappings.json') as f:
#                 mappings = json.load(f)
#             self.plant_type_classes = mappings['plant_types']
#             self.disease_classes = mappings['disease_classes']
#             for plant_type in self.plant_type_classes:
#                 model_file = model_path / f'{plant_type}_disease_classifier.h5'
#                 if model_file.exists():
#                     self.disease_classifiers[plant_type] = {
#                         'model': tf.keras.models.load_model(model_file),
#                         'class_names': self.disease_classes[plant_type]
#                     }
#             print("Models loaded successfully")
#         except Exception as e:
#             raise ValueError(f"Model loading failed: {str(e)}")

# model_loader = HierarchicalModel()
# model_loader.load_models(MODEL_PATH)

# def preprocess_image(image_data):
#     try:
#         image = Image.open(BytesIO(image_data))
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
#         image = image.resize((256, 256))
#         return np.array(image)
#     except Exception as e:
#         raise ValueError(f"Image processing failed: {str(e)}")


# def internet_connected(timeout=3):
#     """Check if we can reach OpenAI API."""
#     try:
#         requests.head("https://api.openai.com", timeout=timeout)
#         return True
#     except requests.RequestException:
#         return False

# @app.get("/")
# def read_root():
#     return {"message": "Plant Disease Detection API", "mode": MODE}


# @app.post("/predict")
# async def predict(request: Request, file: UploadFile = File(...)):
#     try:
#         image_data = await file.read()

#         # Automatically determine mode
#         if internet_connected():
#             mode = "online"
#         else:
#             mode = "offline"

#         # OFFLINE MODE (same as before)
#         if mode == "offline":
#             image_array = preprocess_image(image_data)
#             image_batch = np.expand_dims(image_array, axis=0)

#             plant_predictions = model_loader.plant_classifier.predict(image_batch)
#             plant_type_idx = np.argmax(plant_predictions[0])
#             plant_confidence = plant_predictions[0][plant_type_idx]

#             result = {
#                 "mode": "offline",
#                 "plant_type": "unknown",
#                 "plant_confidence": float(plant_confidence),
#                 "disease": "unknown",
#                 "disease_confidence": 0.0,
#                 "is_healthy": False,
#                 "message": "Offline mode - no AI advice",
#                 "treatment": None
#             }

#             if plant_confidence >= PLANT_TYPE_CONFIDENCE_THRESHOLD:
#                 predicted_plant = model_loader.plant_type_classes[plant_type_idx]
#                 result["plant_type"] = predicted_plant

#                 if predicted_plant in model_loader.disease_classifiers:
#                     disease_model = model_loader.disease_classifiers[predicted_plant]['model']
#                     disease_classes = model_loader.disease_classifiers[predicted_plant]['class_names']
#                     disease_predictions = disease_model.predict(image_batch)
#                     disease_idx = np.argmax(disease_predictions[0])
#                     disease_confidence = disease_predictions[0][disease_idx]

#                     if disease_confidence >= DISEASE_CONFIDENCE_THRESHOLD:
#                         predicted_disease = disease_classes[disease_idx]
#                         result.update({
#                             "disease": predicted_disease,
#                             "disease_confidence": float(disease_confidence),
#                             "is_healthy": "healthy" in predicted_disease.lower(),
#                             "message": f"Disease detected: {predicted_disease} (offline mode)"
#                         })
#             return result

#         # ONLINE MODE - Return JSON + Base64 audio
#         elif mode == "online":
#             if not internet_connected():
#                 raise HTTPException(status_code=503, detail="No internet connection for online mode")

#             temp_image_path = "temp_upload.jpg"
#             with open(temp_image_path, "wb") as f:
#                 f.write(image_data)

#             treatment_json = analyze_image_with_openai(
#                 image_path=temp_image_path,
#                 result_text="Analyze and return disease prediction + treatment advice in JSON format"
#             )

#             audio_url = None
#             if treatment_json.get("voice_output"):  # voice bytes exist
#                  # Save voice to static file
#                 audio_bytes = treatment_json["voice_output"]
#                 filename = f"audio_{uuid.uuid4().hex}.mp3"
#                 filepath = f"static/{filename}"
#                 os.makedirs("static", exist_ok=True)
#                 with open(filepath, "wb") as f:
#                     f.write(audio_bytes)

#                 base_url = str(request.base_url).rstrip("/")
#                 audio_url = f"{base_url}/static/{filename}"
#                 del treatment_json["voice_output"]

#             return {
#                 "mode": "online",
#                 "analysis": treatment_json,
#                 "audio_url":audio_url # 🔹 JSON always includes audio (if available)
#             }

#         else:
#             raise HTTPException(status_code=400, detail="Invalid MODE setting. Use 'offline' or 'online'.")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
import uuid
from io import BytesIO
import numpy as np
from io import BytesIO
from PIL import Image
import os
import tensorflow as tf
from pathlib import Path
import json
import uvicorn
import socket
import os
from fastapi.middleware.cors import CORSMiddleware
from utils import analyze_image_with_openai, SYSTEM_PROMPT, VECTOR_STORE_ID
from fastapi.staticfiles import StaticFiles
import base64
import requests

# MODEL_PATH = Path("hierarchical_models_mobilenetv2")
MODEL_PATH = Path("hierarchical_models_v1")
DISEASE_CONFIDENCE_THRESHOLD = 0.5
PLANT_TYPE_CONFIDENCE_THRESHOLD = 0.6

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Mode can be "offline" or "online"
MODE = os.getenv("MODE", "offline").lower()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

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
        self.models_loaded = False

    def load_models(self, model_path):
        try:
            print(f"TensorFlow version: {tf.__version__}")
            print(f"Loading models from: {model_path}")
            
            # Check if model files exist
            plant_model_path = model_path / 'plant_type_classifier.keras'
            class_mappings_path = model_path / 'class_mappings.json'
            
            if not plant_model_path.exists():
                raise FileNotFoundError(f"Plant classifier model not found at: {plant_model_path}")
            
            if not class_mappings_path.exists():
                raise FileNotFoundError(f"Class mappings file not found at: {class_mappings_path}")
            
            print("Loading plant type classifier...")
            # Try different loading approaches
            try:
                # First try: Load with compile=False
                self.plant_classifier = tf.keras.models.load_model(
                    plant_model_path,
                    #compile=False
                )
            except Exception as e1:
                print(f"First loading attempt failed: {e1}")
                try:
                    # Second try: Load with custom objects
                    self.plant_classifier = tf.keras.models.load_model(
                        plant_model_path,
                        compile=False,
                        custom_objects=None
                    )
                except Exception as e2:
                    print(f"Second loading attempt failed: {e2}")
                    # Third try: Load with safe mode
                    self.plant_classifier = tf.keras.models.load_model(
                        plant_model_path,
                        compile=False,
                        safe_mode=False
                    )
            
            print("Plant classifier loaded successfully")
            
            # Load class mappings
            with open(class_mappings_path) as f:
                mappings = json.load(f)
            
            self.plant_type_classes = mappings['plant_types']
            self.disease_classes = mappings['disease_classes']
            
            print(f"Plant types: {self.plant_type_classes}")
            
            # Load disease classifiers
            for plant_type in self.plant_type_classes:
                model_file = model_path / f'{plant_type}_disease_classifier.keras'
                if model_file.exists():
                    print(f"Loading disease classifier for {plant_type}...")
                    try:
                        disease_model = tf.keras.models.load_model(
                            model_file,
                            compile=False
                        )
                        self.disease_classifiers[plant_type] = {
                            'model': disease_model,
                            'class_names': self.disease_classes[plant_type]
                        }
                        print(f"Disease classifier for {plant_type} loaded successfully")
                    except Exception as e:
                        print(f"Failed to load disease classifier for {plant_type}: {e}")
                        # Continue loading other models
                        continue
                else:
                    print(f"Disease classifier for {plant_type} not found at {model_file}")
            
            self.models_loaded = True
            print("Models loaded successfully")
            
        except Exception as e:
            print(f"Model loading failed: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            self.models_loaded = False
            # Don't raise the error, allow the app to start in online-only mode
            print("Starting in online-only mode due to model loading failure")

# Initialize model loader
model_loader = HierarchicalModel()

# Try to load models, but don't fail if they can't be loaded
try:
    model_loader.load_models(MODEL_PATH)
except Exception as e:
    print(f"Failed to load models during startup: {e}")
    print("Application will run in online-only mode")

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
    return {
        "message": "Plant Disease Detection API", 
        "mode": MODE,
        "models_loaded": model_loader.models_loaded,
        "tensorflow_version": tf.__version__
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_loader.models_loaded,
        "available_modes": ["online"] if not model_loader.models_loaded else ["offline", "online"],
        "tensorflow_version": tf.__version__
    }

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        # Check if models are available for offline mode
        if not model_loader.models_loaded:
            print("Models not loaded, forcing online mode")
            mode = "online"
        else:
            # Automatically determine mode
            if internet_connected():
                mode = "online"
            else:
                mode = "offline"

        # OFFLINE MODE
        if mode == "offline":
            if not model_loader.models_loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Offline mode not available - models failed to load"
                )
                
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

        # ONLINE MODE - Return JSON + Base64 audio
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

            audio_url = None
            if treatment_json.get("voice_output"):  # voice bytes exist
                 # Save voice to static file
                audio_bytes = treatment_json["voice_output"]
                filename = f"audio_{uuid.uuid4().hex}.mp3"
                filepath = f"static/{filename}"
                os.makedirs("static", exist_ok=True)
                with open(filepath, "wb") as f:
                    f.write(audio_bytes)

                base_url = str(request.base_url).rstrip("/")
                audio_url = f"{base_url}/static/{filename}"
                del treatment_json["voice_output"]

            return {
                "mode": "online",
                "analysis": treatment_json,
                "audio_url": audio_url,
                "models_available": model_loader.models_loaded
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid MODE setting. Use 'offline' or 'online'.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)