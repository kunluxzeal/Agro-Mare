from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from pathlib import Path
import json
import uvicorn
import google.generativeai as genai
import re
from fastapi.middleware.cors import CORSMiddleware

# Configure Gemini AI
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-2.5-flash')

app = FastAPI()

# CORS Configuration
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model configuration
MODEL_PATH = Path("../training/hierarchical_models_mobilenetv2")  # Directory where .h5 models are saved
DISEASE_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence for disease predictions
PLANT_TYPE_CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for plant type predictions

class HierarchicalModel:
    def __init__(self):
        self.plant_classifier = None
        self.disease_classifiers = {}
        self.plant_type_classes = []
        self.disease_classes = {}
        
    def load_models(self, model_path):
        """Load all models from .h5 files and class mappings"""
        try:
            # Load plant classifier
            self.plant_classifier = tf.keras.models.load_model(
                model_path / 'plant_type_classifier.h5'
            )
            
            # Load class mappings
            with open(model_path / 'class_mappings.json') as f:
                mappings = json.load(f)
            
            self.plant_type_classes = mappings['plant_types']
            self.disease_classes = mappings['disease_classes']
            
            # Load disease classifiers
            for plant_type in self.plant_type_classes:
                model_file = model_path / f'{plant_type}_disease_classifier.h5'
                if model_file.exists():
                    self.disease_classifiers[plant_type] = {
                        'model': tf.keras.models.load_model(model_file),
                        'class_names': self.disease_classes[plant_type]
                    }
            
            print("Models loaded successfully")
            print("Plant types:", self.plant_type_classes)
            print("Disease classifiers loaded for:", list(self.disease_classifiers.keys()))
            
        except Exception as e:
            raise ValueError(f"Model loading failed: {str(e)}")

# Initialize model loader
model_loader = HierarchicalModel()
model_loader.load_models(MODEL_PATH)

def preprocess_image(image_data):
    try:
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        image_array = np.array(image)
        return image_array
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")

async def get_llm_advice(plant: str, disease: str) -> str:
    if "healthy" in disease.lower():
        return f"{plant} appears healthy! No treatment needed."
    elif "unknown" in disease.lower():
        return "Could not confidently identify the condition."
    
    try:
        prompt = f"""Provide a concise treatment plan for {disease} in {plant} plants as JSON:
        {{
            "IMMEDIATE_ACTION": "...",
            "ORGANIC_TREATMENT": "...",
            "CHEMICAL_TREATMENT": "...",
            "PREVENTION": "..."
        }}"""
        
        response = await model.generate_content_async(prompt)
        response_text = response.text.strip()
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"error": "Could not parse treatment advice"}
    
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return {"error": f"Could not generate treatment advice for {disease}"}

@app.get("/")
def read_root():
    return {"message": "Plant Disease Detection API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Preprocess image
        image_data = await file.read()
        image_array = preprocess_image(image_data)
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Plant type prediction
        plant_predictions = model_loader.plant_classifier.predict(image_batch)
        plant_type_idx = np.argmax(plant_predictions[0])
        plant_confidence = plant_predictions[0][plant_type_idx]
        
        result = {
            'plant_type': 'unknown',
            'plant_confidence': float(plant_confidence),
            'disease': 'unknown',
            'disease_confidence': 0.0,
            'is_healthy': False,
            'message': ''
        }
        
        # Check if plant type confidence meets threshold
        if plant_confidence >= PLANT_TYPE_CONFIDENCE_THRESHOLD:
            predicted_plant = model_loader.plant_type_classes[plant_type_idx]
            result['plant_type'] = predicted_plant
            result['message'] = f"Plant identified as {predicted_plant}"
            
            # Proceed with disease prediction if plant is identified and disease classifier exists
            if predicted_plant in model_loader.disease_classifiers:
                disease_model = model_loader.disease_classifiers[predicted_plant]['model']
                disease_classes = model_loader.disease_classifiers[predicted_plant]['class_names']
                
                disease_predictions = disease_model.predict(image_batch)
                disease_idx = np.argmax(disease_predictions[0])
                disease_confidence = disease_predictions[0][disease_idx]
                
                # Check disease confidence threshold
                if disease_confidence >= DISEASE_CONFIDENCE_THRESHOLD:
                    predicted_disease = disease_classes[disease_idx]
                    result.update({
                        'disease': predicted_disease,
                        'disease_confidence': float(disease_confidence),
                        'is_healthy': 'healthy' in predicted_disease.lower()
                    })
                    
                    # Get treatment advice if not healthy
                    if not result['is_healthy']:
                        result['treatment'] = await get_llm_advice(
                            plant=predicted_plant,
                            disease=predicted_disease
                        )
                        result['message'] = f"Disease detected: {predicted_disease}"
                    else:
                        result['message'] = f"{predicted_plant} appears healthy!"
                else:
                    result['message'] = f"Plant identified as {predicted_plant}, but disease prediction confidence too low"
            else:
                result['message'] = f"Plant identified as {predicted_plant}, but no disease classifier available"
        else:
            result['message'] = "Not sure of the image - plant type confidence too low"
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)