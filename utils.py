import openai
import os
from dotenv import load_dotenv
import base64
import json
import re


load_dotenv()

client = openai.Client(
    api_key=os.getenv("OPENAI_API_KEY", "")
)


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


# Vector store ID for file searching
VECTOR_STORE_ID = "vs_689cb1ad26948191ad81f68542ff8a04"

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
        return json.loads(cleaned_text)
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
