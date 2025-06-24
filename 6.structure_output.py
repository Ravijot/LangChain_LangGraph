import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import List, Optional
import json

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai")

class ResponseFormatter(BaseModel):
    emotion : str = Field(description="The emotion conveyed in the response e.g. happy, sad, angry, etc.")
    confidence_score : float = Field(description="The confidence score of the emotion conveyed in the response, between 0 and 1.")
    
model_with_tools = model.bind_tools([ResponseFormatter])
response = model_with_tools.invoke("Hurrah! I just won the lottery!")
print(response.additional_kwargs['tool_calls'][0]['function']['arguments'])
