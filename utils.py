import openai
import os
from dotenv import load_dotenv
import base64
import json
import re
from elevenlabs import ElevenLabs 
from pydub import AudioSegment
import tempfile
import requests
from io import BytesIO

load_dotenv()

# Initialize API keys
client = openai.Client(api_key=os.getenv("OPENAI_API_KEY", ""))


stores = client.vector_stores.list()
for vs in stores.data:
    print(vs.id, vs.name)

# ElevenLabs client
tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY", ""))


# Voice IDs for different scenarios
VOICE_IDS = {
    "default": os.getenv("ELEVEN_LABS_DEFAULT_VOICE", "21m00Tcm4TlvDq8ikWAM"),  # Rachel voice
    "alert": os.getenv("ELEVEN_LABS_ALERT_VOICE", "AZnzlk1XvdvUeBnXmlld"),     # Domi voice
    "info": os.getenv("ELEVEN_LABS_INFO_VOICE", "EXAVITQu4vr4xnSDxMaL"),        # Bella voice
    "olufunmilola" : os.getenv("ELEVEN_LABS_OLUFUNMILOLA_VOICE", "9Dbo4hEvXQ5l7MXGZFQA") # Olufunmilola voice
}


SYSTEM_PROMPT = """
You are a plant disease analysis assistant.
You can analyze images and also use the file_search tool if needed.

IMPORTANT RULES:
1. ALWAYS analyze the provided plant leaf image yourself first.
2. You MAY use file_search only to improve treatment recommendations, but do not skip image analysis.
3. Your final output MUST always be a complete JSON object, containing:
   - plant_type: name of the plant
   - plant_confidence: your confidence score between 0 and 1
   - disease: name of the detected disease, or 'healthy'
   - disease_confidence: your confidence score between 0 and 1
   - is_healthy: true if no disease, false otherwise
   - message: short status message
   - treatment: object with:
       cultural: cultural treatment steps
       chemical: chemical treatment steps
       preventive: preventive steps

No explanations, no markdown, no text outside JSON.
"""


# # Vector store ID for file searching
# VECTOR_STORE_ID = "vs_689cb1ad26948191ad81f68542ff8a04"

def get_or_create_vector_store():
    # 1. List existing vector stores
    stores = client.vector_stores.list()
    if stores.data:
        # return the first one
        return stores.data[0].id

    # 2. If none exist, create one
    new_store = client.vector_stores.create(name="plant_disease")
    return new_store.id

# Dynamically fetch/create
VECTOR_STORE_ID = get_or_create_vector_store()

def analyze_image_with_openai(image_path: str, result_text: str = "Analyze this plant leaf image and return the results in JSON format.") -> dict:
    """
    Passes the initial image and result text to OpenAI for further analysis.
    Returns the model's response.
    """

    # Read image and encode as base64 (if needed)
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # prompt = f"Result: {result_text}\n\n"
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
            "role": "developer",
            "content": [
                {
                "type": "input_text",
                "text": SYSTEM_PROMPT
                }
            ]
            },
            {"role": "user", "content": [
                {
                    "type": "input_text",
                    "text":  result_text
                    
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_base64}"
                }
            ]}
        ],
        tools=[
                    {
            "type": "file_search",
            "vector_store_ids": [VECTOR_STORE_ID]
            }
        ],

    )

    
    try:
        raw_text = extract_text_from_response(response)
        cleaned_text = clean_json_string(raw_text)
        result = json.loads(cleaned_text)
        
        voice_message = f"I detected a {result['plant_type']} plant. "

        if result['is_healthy']:
            voice_message += "The plant appears to be healthy."
             # Append treatment steps safely
            for section in ["cultural", "chemical", "preventive"]:
                steps = result['treatment'].get(section)
                if steps:
                    if isinstance(steps, list):
                        voice_message += " ".join(steps) + ". "
                    else:
                        voice_message += str(steps) + ". "

            voice_type = "olufunmilola"
        else:
            voice_message += f"I found {result['disease']}. "

            # Append treatment steps safely
            for section in ["cultural", "chemical", "preventive"]:
                steps = result['treatment'].get(section)
                if steps:
                    if isinstance(steps, list):
                        voice_message += " ".join(steps) + ". "
                    else:
                        voice_message += str(steps) + ". "
            
            voice_type = "olufunmilola"

        # Generate voice output
        result['voice_output'] = generate_voice_output(voice_message, voice_type)
        
        return result
    except Exception as e:
        return {"error": f"Could not parse OpenAI response: {e}", "raw_text": raw_text if 'raw_text' in locals() else None}



# # Example usage:
# analysis = analyze_image_with_openai("path/to/image.jpg", "Detected: Leaf spot")
# print(analysis)


def extract_text_from_response(response):
    text_segments = []
    for output in response.output:
        for content in output.content:
            if content.type == "output_text":
                text_segments.append(content.text)
    return "".join(text_segments).strip()



def clean_json_string(raw_text):
    # Remove code fences like ```json ... ```
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return cleaned.strip()



def generate_voice_output(text: str, voice_type: str = "default") -> bytes:
    """
    Generate voice output using Eleven Labs API and return full MP3 bytes.
    """
    voice_id = VOICE_IDS.get(voice_type, VOICE_IDS["olufunmilola"])
    api_key = os.getenv("ELEVEN_LABS_API_KEY")

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json"
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.7,
            "similarity_boost": 0.7
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.content  # 🔹 FULL mp3 file
    except Exception as e:
        print(f"Voice generation failed: {str(e)}")
        return None

