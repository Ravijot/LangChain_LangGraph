from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."

flight_assistant = create_react_agent(
    model=model,
    tools=[book_flight],
    prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model=model,
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)

supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        "You manage a hotel booking assistant and a"
        "flight booking assistant. Assign work to them."
    )
).compile()
""" 
This code will create a graph of above code and save as graph_output.png in local folder
"""
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = supervisor.get_graph().draw_mermaid_png()

# Save it to a file
with open("supervisor_api.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("supervisor_api.png")
img.show()

# for chunk in supervisor.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         ]
#     }
# ):
#     print(chunk)
#     print("\n")

response = supervisor.invoke({
        "messages": [
            {
                "role": "user",
                "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
            }
        ]
    })  
for message in response['messages']:
    if isinstance(message, HumanMessage):
        print("[Human Message]")
    elif isinstance(message, AIMessage):
        print("[AI Message]")
    elif isinstance(message, ToolMessage):
        print("[Tool Message]")
    else:
        print("[Unknown Message Type]")
    
    print(message.content)
    print("-" * 40)  # optional separator