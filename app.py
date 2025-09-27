

import asyncio
import concurrent.futures
import threading
from functools import wraps
import uuid
import tempfile
import os
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path
import json
import requests
from io import BytesIO

# Import your existing modules
from utils import analyze_image_with_openai_chat
from AudioToS3 import UploadToS3

# Configuration
MODEL_PATH = Path("hierarchical_models_v2")
DISEASE_CONFIDENCE_THRESHOLD = 0.5
PLANT_TYPE_CONFIDENCE_THRESHOLD = 0.6
MAX_WORKERS = 4  # Adjust based on your server capacity

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread-local storage for models
thread_local_data = threading.local()

class ThreadSafeModelLoader:
    """Thread-safe model loader that creates separate model instances per thread"""
    
    def __init__(self):
        self.plant_type_classes = []
        self.disease_classes = {}
        self.models_loaded = False
        self.model_path = None
        self._lock = threading.Lock()
        
    def load_class_mappings(self, model_path):
        """Load class mappings once (thread-safe)"""
        with self._lock:
            if self.models_loaded:
                return True
                
            class_mappings_path = model_path / 'class_mappings.json'
            if not class_mappings_path.exists():
                print(f"Class mappings file not found at: {class_mappings_path}")
                return False
                
            try:
                with open(class_mappings_path) as f:
                    mappings = json.load(f)
                
                self.plant_type_classes = mappings['plant_types']
                self.disease_classes = mappings['disease_classes']
                self.model_path = model_path
                self.models_loaded = True
                print(f"Class mappings loaded: {self.plant_type_classes}")
                return True
            except Exception as e:
                print(f"Failed to load class mappings: {e}")
                return False
    
    def get_thread_models(self):
        """Get or create models for current thread"""
        if not hasattr(thread_local_data, 'plant_classifier'):
            if not self.models_loaded:
                return None, {}
                
            print(f"Loading models for thread {threading.current_thread().ident}")
            
            # Load plant classifier for this thread
            plant_model_path = self.model_path / 'plant_type_classifier.keras'
            if plant_model_path.exists():
                try:
                    thread_local_data.plant_classifier = tf.keras.models.load_model(
                        plant_model_path, compile=False
                    )
                    print("Plant classifier loaded for thread")
                except Exception as e:
                    print(f"Failed to load plant classifier for thread: {e}")
                    thread_local_data.plant_classifier = None
            else:
                thread_local_data.plant_classifier = None
            
            # Load disease classifiers for this thread
            thread_local_data.disease_classifiers = {}
            for plant_type in self.plant_type_classes:
                model_file = self.model_path / f'{plant_type}_disease_classifier.keras'
                if model_file.exists():
                    try:
                        disease_model = tf.keras.models.load_model(
                            model_file, compile=False
                        )
                        thread_local_data.disease_classifiers[plant_type] = {
                            'model': disease_model,
                            'class_names': self.disease_classes[plant_type]
                        }
                        print(f"Disease classifier for {plant_type} loaded for thread")
                    except Exception as e:
                        print(f"Failed to load disease classifier for {plant_type} in thread: {e}")
        
        return getattr(thread_local_data, 'plant_classifier', None), \
               getattr(thread_local_data, 'disease_classifiers', {})

# Global thread-safe model loader
model_loader = ThreadSafeModelLoader()

# Initialize class mappings at startup
model_loader.load_class_mappings(MODEL_PATH)

# Thread pool for CPU-bound tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

def run_in_thread_pool(func):
    """Decorator to run function in thread pool"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args, **kwargs)
    return wrapper

def preprocess_image(image_data: bytes) -> np.ndarray:
    """Preprocess image data"""
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

@run_in_thread_pool
def predict_offline(image_data: bytes):
    """Run offline prediction in thread pool"""
    try:
        # Get thread-specific models
        plant_classifier, disease_classifiers = model_loader.get_thread_models()
        
        if plant_classifier is None:
            raise ValueError("Plant classifier not available")
            
        image_array = preprocess_image(image_data)
        image_batch = np.expand_dims(image_array, axis=0)

        # Predict plant type
        plant_predictions = plant_classifier.predict(image_batch)
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

            if predicted_plant in disease_classifiers:
                disease_model = disease_classifiers[predicted_plant]['model']
                disease_classes = disease_classifiers[predicted_plant]['class_names']
                
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
    except Exception as e:
        raise Exception(f"Offline prediction failed: {str(e)}")

async def predict_online(image_data: bytes):
    """Run online prediction asynchronously"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            temp_image_path = tmp.name

        # try:
        #     # Run OpenAI analysis in thread pool to avoid blocking
        #     loop = asyncio.get_event_loop()
        #     treatment_json = await loop.run_in_executor(
        #         executor,
        #         analyze_image_with_openai,
        #         temp_image_path,
        #         "Analyze and return disease prediction + treatment advice in JSON format"
        #     )

        try:
            # Run OpenAI analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            treatment_json = await loop.run_in_executor(
                executor,
                analyze_image_with_openai_chat,
                temp_image_path,
                "Analyze and return disease prediction + treatment advice in JSON format"
            )

            # Validate the response structure
            if not isinstance(treatment_json, dict):
                treatment_json = {"error": "Invalid response format from OpenAI", "raw_text": str(treatment_json)}


            audio_url = None
            if treatment_json.get("voice_output"):
                # Upload audio to S3 in thread pool
                audio_url = await loop.run_in_executor(
                    executor,
                    UploadToS3,
                    treatment_json["voice_output"]
                )
                del treatment_json["voice_output"]

            return {
                "mode": "online",
                "analysis": treatment_json,
                "audio_url": audio_url,
                "models_available": model_loader.models_loaded
            }

        finally:
            # Clean up temp file
            try:
                os.remove(temp_image_path)
            except:
                pass

    except Exception as e:
        raise Exception(f"Online prediction failed: {str(e)}")

def internet_connected(timeout=3) -> bool:
    """Check if we can reach OpenAI API"""
    try:
        requests.head("https://api.openai.com", timeout=timeout)
        return True
    except requests.RequestException:
        return False

@app.get("/")
def read_root():
    return {
        "message": "Plant Disease Detection API - Concurrent Version",
        "models_loaded": model_loader.models_loaded,
        "tensorflow_version": tf.__version__,
        "max_workers": MAX_WORKERS
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": model_loader.models_loaded,
        "available_modes": ["online"] if not model_loader.models_loaded else ["offline", "online"],
        "tensorflow_version": tf.__version__,
        "active_threads": threading.active_count()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image data
        image_data = await file.read()

        # Determine mode
        if not model_loader.models_loaded:
            mode = "online"
        else:
            mode = "online" if internet_connected() else "offline"

        # Execute prediction based on mode
        if mode == "offline":
            if not model_loader.models_loaded:
                raise HTTPException(
                    status_code=503, 
                    detail="Offline mode not available - models failed to load"
                )
            
            # Run offline prediction in thread pool
            result = await predict_offline(image_data)
            return result

        elif mode == "online":
            if not internet_connected():
                raise HTTPException(status_code=503, detail="No internet connection for online mode")
            
            # Run online prediction asynchronously
            result = await predict_online(image_data)
            return result

        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """Clean up thread pool on shutdown"""
    executor.shutdown(wait=True)