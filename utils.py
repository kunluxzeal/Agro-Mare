import openai

# Dummy system prompt (replace with your own)
SYSTEM_PROMPT = "You are a helpful assistant for plant disease analysis."

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
    # You may need to encode image_bytes to base64 if required by OpenAI
    # import base64
    # image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    prompt = f"{SYSTEM_PROMPT}\n\nResult: {result_text}\n\nAnalyze the image and result."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Example usage:
# analysis = analyze_image_with_openai("path/to/image.jpg", "Detected: Leaf spot")
# print(analysis)
