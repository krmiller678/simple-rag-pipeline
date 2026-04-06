from typing import List
from interface.base_datastore import BaseDatastore, DataItem
import lancedb
from lancedb.table import Table
import pyarrow as pa
from google import genai
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv()

def invoke_ai(system_message: str, user_message: str) -> str:
    """Invokes Gemini using the GenAI SDK."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Gemini uses a 'contents' list where the system instruction is a separate param
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        config={'system_instruction': system_message},
        contents=user_message
    )
    return response.text