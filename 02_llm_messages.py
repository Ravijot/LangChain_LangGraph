import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

messages = [
    SystemMessage(content="You are a customer support assistant of a tech company. Expert in troubleshooting and providing solutions."),
    HumanMessage(content="Hello, I am having trouble with my laptop. It won't turn on."),
]

ai_message = AIMessage(content=model.invoke(messages).content)
print(ai_message)

# for chunk in model.stream(messages):
#     print(chunk.content, end="", flush=True)



