import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

print(model.invoke("Tell me a joke about AI."))