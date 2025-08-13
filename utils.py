import openai
import os
from dotenv import load_dotenv
import base64


load_dotenv()

client = openai.Client(
    api_key=os.getenv("OPENAI_API_KEY", "")
)


SYSTEM_PROMPT = "You are a helpful assistant for plant disease analysis. you will analyze images and provide treatment advice 

# Vector store ID for file searching
VECTOR_STORE_ID = "vs_689cb1ad26948191ad81f68542ff8a04"

def analyze_image_with_openai(image_path, result_text):
    """
    Passes the initial image and result text to OpenAI for further analysis.
    Returns the model's response.
    """

    # Read image and encode as base64 (if needed)
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    prompt = f"Result: {result_text}\n\n"
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
                    "text": prompt
                },
                {
                    "type": "input_image",
                    "image": f"data:image/jpeg;base64,{image_base64}"
                }
            ]}
        ],
        tools=[
                    {
            "type": "file_search",
            "vector_store_ids": [
                "vs_689cb1ad26948191ad81f68542ff8a04"
            ]
            }
        ],

    )
    return response.text

# Example usage:
# analysis = analyze_image_with_openai("path/to/image.jpg", "Detected: Leaf spot")
# print(analysis)
