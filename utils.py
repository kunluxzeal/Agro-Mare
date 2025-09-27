# import openai
# import os
# from dotenv import load_dotenv
# import base64
# import json
# import re
# from elevenlabs import ElevenLabs 
# from pydub import AudioSegment
# import tempfile
# import requests
# from io import BytesIO

# load_dotenv()

# # Initialize API keys
# client = openai.Client(api_key=os.getenv("OPENAI_API_KEY", ""))


# stores = client.vector_stores.list()
# for vs in stores.data:
#     print(vs.id, vs.name)

# # ElevenLabs client
# tts_client = ElevenLabs(api_key=os.getenv("ELEVEN_LABS_API_KEY", ""))


# # Voice IDs for different scenarios
# VOICE_IDS = {
#     "default": os.getenv("ELEVEN_LABS_DEFAULT_VOICE", "21m00Tcm4TlvDq8ikWAM"),  # Rachel voice
#     "alert": os.getenv("ELEVEN_LABS_ALERT_VOICE", "AZnzlk1XvdvUeBnXmlld"),     # Domi voice
#     "info": os.getenv("ELEVEN_LABS_INFO_VOICE", "EXAVITQu4vr4xnSDxMaL"),        # Bella voice
#     "olufunmilola" : os.getenv("ELEVEN_LABS_OLUFUNMILOLA_VOICE", "9Dbo4hEvXQ5l7MXGZFQA") # Olufunmilola voice
# }


# SYSTEM_PROMPT = """
# You are a plant disease analysis assistant.
# You can analyze images and also use the file_search tool if needed.

# IMPORTANT RULES:
# 1. ALWAYS analyze the provided plant leaf image yourself first.
# 2. You MAY use file_search only to improve treatment recommendations, but do not skip image analysis.
# 3. Your final output MUST always be a complete JSON object, containing:
#    - plant_type: name of the plant
#    - plant_confidence: your confidence score between 0 and 1
#    - disease: name of the detected disease, or 'healthy'
#    - disease_confidence: your confidence score between 0 and 1
#    - is_healthy: true if no disease, false otherwise
#    - message: short status message
#    - treatment: object with:
#        cultural: cultural treatment steps
#        chemical: chemical treatment steps
#        preventive: preventive steps

# No explanations, no markdown, no text outside JSON.
# """


# # # Vector store ID for file searching
# # VECTOR_STORE_ID = "vs_689cb1ad26948191ad81f68542ff8a04"

# def get_or_create_vector_store():
#     # 1. List existing vector stores
#     stores = client.vector_stores.list()
#     if stores.data:
#         # return the first one
#         return stores.data[0].id

#     # 2. If none exist, create one
#     new_store = client.vector_stores.create(name="plant_disease")
#     return new_store.id

# # Dynamically fetch/create
# VECTOR_STORE_ID = get_or_create_vector_store()

# def analyze_image_with_openai(image_path: str, result_text: str = "Analyze this plant leaf image and return the results in JSON format.") -> dict:
#     """
#     Passes the initial image and result text to OpenAI for further analysis.
#     Returns the model's response.
#     """

#     # Read image and encode as base64 (if needed)
#     with open(image_path, "rb") as img_file:
#         image_bytes = img_file.read()

#     image_base64 = base64.b64encode(image_bytes).decode('utf-8')

#     # prompt = f"Result: {result_text}\n\n"
#     response = client.responses.create(
#         model="gpt-4o",
#         input=[
#             {
#             "role": "developer",
#             "content": [
#                 {
#                 "type": "input_text",
#                 "text": SYSTEM_PROMPT
#                 }
#             ]
#             },
#             {"role": "user", "content": [
#                 {
#                     "type": "input_text",
#                     "text":  result_text
                    
#                 },
#                 {
#                     "type": "input_image",
#                     "image_url": f"data:image/jpeg;base64,{image_base64}"
#                 }
#             ]}
#         ],
#         tools=[
#                     {
#             "type": "file_search",
#             "vector_store_ids": [VECTOR_STORE_ID]
#             }
#         ],

#     )

    
#     try:
#         raw_text = extract_text_from_response(response)
#         cleaned_text = clean_json_string(raw_text)
#         result = json.loads(cleaned_text)
        
#         voice_message = f"I detected a {result['plant_type']} plant. "

#         if result['is_healthy']:
#             voice_message += "The plant appears to be healthy."
#              # Append treatment steps safely
#             for section in ["cultural", "chemical", "preventive"]:
#                 steps = result['treatment'].get(section)
#                 if steps:
#                     if isinstance(steps, list):
#                         voice_message += " ".join(steps) + ". "
#                     else:
#                         voice_message += str(steps) + ". "

#             voice_type = "olufunmilola"
#         else:
#             voice_message += f"I found {result['disease']}. "

#             # Append treatment steps safely
#             for section in ["cultural", "chemical", "preventive"]:
#                 steps = result['treatment'].get(section)
#                 if steps:
#                     if isinstance(steps, list):
#                         voice_message += " ".join(steps) + ". "
#                     else:
#                         voice_message += str(steps) + ". "
            
#             voice_type = "olufunmilola"

#         # Generate voice output
#         result['voice_output'] = generate_voice_output(voice_message, voice_type)
        
#         return result
#     except Exception as e:
#         return {"error": f"Could not parse OpenAI response: {e}", "raw_text": raw_text if 'raw_text' in locals() else None}



# # # Example usage:
# # analysis = analyze_image_with_openai("path/to/image.jpg", "Detected: Leaf spot")
# # print(analysis)


# def extract_text_from_response(response):
#     text_segments = []
#     for output in response.output:
#         for content in output.content:
#             if content.type == "output_text":
#                 text_segments.append(content.text)
#     return "".join(text_segments).strip()



# def clean_json_string(raw_text):
#     # Remove code fences like ```json ... ```
#     cleaned = re.sub(r"^```(json)?", "", raw_text.strip(), flags=re.IGNORECASE)
#     cleaned = re.sub(r"```$", "", cleaned.strip())
#     return cleaned.strip()



# def generate_voice_output(text: str, voice_type: str = "default") -> bytes:
#     """
#     Generate voice output using Eleven Labs API and return full MP3 bytes.
#     """
#     voice_id = VOICE_IDS.get(voice_type, VOICE_IDS["olufunmilola"])
#     api_key = os.getenv("ELEVEN_LABS_API_KEY")

#     url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
#     headers = {
#         "xi-api-key": api_key,
#         "Accept": "audio/mpeg",
#         "Content-Type": "application/json"
#     }
#     data = {
#         "text": text,
#         "model_id": "eleven_multilingual_v2",
#         "voice_settings": {
#             "stability": 0.7,
#             "similarity_boost": 0.7
#         }
#     }

#     try:
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()
#         return response.content  # 🔹 FULL mp3 file
#     except Exception as e:
#         print(f"Voice generation failed: {str(e)}")
#         return None

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

# List vector stores for debugging
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
    "olufunmilola": os.getenv("ELEVEN_LABS_OLUFUNMILOLA_VOICE", "9Dbo4hEvXQ5l7MXGZFQA")  # Olufunmilola voice
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

def get_or_create_vector_store():
    """Get or create vector store for file search"""
    stores = client.vector_stores.list()
    if stores.data:
        return stores.data[0].id

    new_store = client.vector_stores.create(name="plant_disease")
    return new_store.id

# Dynamically fetch/create
VECTOR_STORE_ID = get_or_create_vector_store()

def extract_text_from_response(response):
    """
    Extract text from OpenAI Responses API response object.
    Handles different content types including tool calls.
    """
    try:
        text_segments = []
        
        # Check if response has output attribute
        if hasattr(response, 'output') and response.output:
            for output in response.output:
                if hasattr(output, 'content') and output.content:
                    for content in output.content:
                        if hasattr(content, 'type'):
                            if content.type == "output_text":
                                text_segments.append(content.text)
                            elif content.type == "tool_call":
                                # Handle tool calls - they might contain relevant information
                                if hasattr(content, 'function') and hasattr(content.function, 'arguments'):
                                    text_segments.append(content.function.arguments)
                            elif hasattr(content, 'text'):
                                # Fallback for any content with text attribute
                                text_segments.append(content.text)
        
        # If no text found, try alternative extraction methods
        if not text_segments:
            # Try to access response content directly
            if hasattr(response, 'choices') and response.choices:
                for choice in response.choices:
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        text_segments.append(choice.message.content)
        
        result_text = "".join(text_segments).strip()
        
        # If still no text, return error information
        if not result_text:
            return json.dumps({
                "error": "No text content found in response",
                "response_type": str(type(response)),
                "has_output": hasattr(response, 'output'),
                "output_length": len(response.output) if hasattr(response, 'output') else 0
            })
        
        return result_text
        
    except Exception as e:
        return json.dumps({
            "error": f"Error extracting text from response: {str(e)}",
            "response_type": str(type(response))
        })

def clean_json_string(raw_text):
    """Clean JSON string by removing markdown code fences"""
    try:
        # First check if it's already valid JSON
        json.loads(raw_text)
        return raw_text
    except json.JSONDecodeError:
        pass
    
    # Remove code fences like ```json ... ```
    cleaned = re.sub(r"^```(json)?", "", raw_text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    
    # Try to extract JSON from text if it's mixed content
    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if json_match:
        return json_match.group(0)
    
    return cleaned.strip()

def create_fallback_response(error_message, raw_text=None):
    """Create a fallback response when parsing fails"""
    return {
        "plant_type": "unknown",
        "plant_confidence": 0.0,
        "disease": "analysis_failed",
        "disease_confidence": 0.0,
        "is_healthy": False,
        "message": f"Analysis failed: {error_message}",
        "treatment": {
            "cultural": ["Unable to provide recommendations due to analysis failure"],
            "chemical": [],
            "preventive": []
        },
        "error": error_message,
        "raw_text": raw_text
    }

def analyze_image_with_openai(image_path: str, result_text: str = "Analyze this plant leaf image and return the results in JSON format.") -> dict:
    """
    Analyze plant image using OpenAI Responses API with proper error handling.
    """
    try:
        # Read image and encode as base64
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Make API call using Responses API
        try:
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
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "input_text",
                                "text": result_text
                            },
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        ]
                    }
                ],
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [VECTOR_STORE_ID]
                    }
                ],
            )
        except Exception as api_error:
            print(f"OpenAI API error: {str(api_error)}")
            return create_fallback_response(f"OpenAI API error: {str(api_error)}")

        # Extract text from response
        try:
            raw_text = extract_text_from_response(response)
        except Exception as extract_error:
            print(f"Text extraction error: {str(extract_error)}")
            return create_fallback_response(f"Failed to extract response: {str(extract_error)}")

        # Check if raw_text is already an error response
        if raw_text.strip().startswith('{"error"'):
            try:
                error_data = json.loads(raw_text)
                return create_fallback_response(error_data.get("error", "Unknown extraction error"), raw_text)
            except:
                return create_fallback_response("Failed to parse error response", raw_text)

        # Clean and parse JSON
        try:
            cleaned_text = clean_json_string(raw_text)
            result = json.loads(cleaned_text)
        except json.JSONDecodeError as json_error:
            print(f"JSON parsing error: {str(json_error)}")
            print(f"Raw text: {raw_text[:500]}...")
            return create_fallback_response(f"Could not parse JSON: {str(json_error)}", raw_text)

        # Validate required fields
        required_fields = ['plant_type', 'disease', 'is_healthy', 'treatment']
        for field in required_fields:
            if field not in result:
                print(f"Missing required field: {field}")
                return create_fallback_response(f"Missing required field: {field}", raw_text)

        # Generate voice message
        try:
            voice_message = f"I detected a {result['plant_type']} plant. "

            if result['is_healthy']:
                voice_message += "The plant appears to be healthy."
                voice_type = "olufunmilola"
            else:
                voice_message += f"I found {result['disease']}. "
                voice_type = "olufunmilola"

            # Append treatment steps safely
            if isinstance(result.get('treatment'), dict):
                for section in ["cultural", "chemical", "preventive"]:
                    steps = result['treatment'].get(section, [])
                    if steps:
                        if isinstance(steps, list):
                            voice_message += " ".join(steps[:2]) + ". "  # Limit to first 2 steps
                        else:
                            voice_message += str(steps) + ". "

            # Generate voice output
            voice_output = generate_voice_output(voice_message, voice_type)
            if voice_output:
                result['voice_output'] = voice_output

        except Exception as voice_error:
            print(f"Voice generation error: {str(voice_error)}")
            # Don't fail the whole request for voice errors

        return result

    except FileNotFoundError:
        return create_fallback_response("Image file not found")
    except Exception as e:
        print(f"Unexpected error in analyze_image_with_openai: {str(e)}")
        return create_fallback_response(f"Unexpected error: {str(e)}")

def generate_voice_output(text: str, voice_type: str = "default") -> bytes:
    """
    Generate voice output using Eleven Labs API and return full MP3 bytes.
    """
    try:
        voice_id = VOICE_IDS.get(voice_type, VOICE_IDS["olufunmilola"])
        api_key = os.getenv("ELEVEN_LABS_API_KEY")
        
        if not api_key:
            print("ElevenLabs API key not found")
            return None

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Accept": "audio/mpeg",
            "Content-Type": "application/json"
        }
        data = {
            "text": text[:1000],  # Limit text length
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.7,
                "similarity_boost": 0.7
            }
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.content  # Full mp3 file

    except Exception as e:
        print(f"Voice generation failed: {str(e)}")
        return None

# Alternative function using Chat Completions API (more reliable)
def analyze_image_with_openai_chat(image_path: str, result_text: str = "Analyze this plant leaf image and return the results in JSON format.") -> dict:
    """
    Alternative method using Chat Completions API instead of Responses API.
    This is more reliable and widely supported.
    """
    try:
        # Read image and encode as base64
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()

        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Create messages for Chat Completions API
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]

        # Make API call using Chat Completions
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
            response_format={"type": "json_object"}  # Force JSON response
        )

        # Extract content
        raw_text = response.choices[0].message.content
        result = json.loads(raw_text)

        # Generate voice message (same as above)
        voice_message = f"I detected a {result.get('plant_type', 'unknown')} plant. "
        
        if result.get('is_healthy', False):
            voice_message += "The plant appears to be healthy."
        else:
            voice_message += f"I found {result.get('disease', 'an issue')}. "

        # Generate voice output
        voice_output = generate_voice_output(voice_message, "olufunmilola")
        if voice_output:
            result['voice_output'] = voice_output

        return result

    except Exception as e:
        print(f"Chat API error: {str(e)}")
        return create_fallback_response(f"Chat API error: {str(e)}")