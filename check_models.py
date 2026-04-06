from google import genai
import os
from dotenv import load_dotenv

# Load the GEMINI_API_KEY from your .env file
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("--- All Available Models ---")
models = client.models.list()
for model in models:
    # Use 'model.name' to get the ID for your code
    print(f"ID: {model.name} | Display: {model.display_name}")

print("\n--- Models Specifically for Embedding ---")
# Reset the iterator for a second pass
models = client.models.list()
for model in models:
    # The correct attribute name in the new SDK is 'supported_methods'
    if 'embedContent' in model.supported_methods:
        print(f"✅ FOUND: {model.name}")