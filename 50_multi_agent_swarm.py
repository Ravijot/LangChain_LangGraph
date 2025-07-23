from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_swarm, create_handoff_tool
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)


def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

alice = create_react_agent(
    model,
    [add, create_handoff_tool(agent_name="Bob")],
    prompt="You are Alice, an addition expert.",
    name="Alice",
)

bob = create_react_agent(
    model,
    [create_handoff_tool(agent_name="Alice", description="Transfer to Alice, she can help with math")],
    prompt="You are Bob, you speak like a teacher and can help with math.",
    name="Bob",
)

checkpointer = InMemorySaver()
workflow = create_swarm(
    [alice, bob],
    default_active_agent="Alice"
)
app = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
turn_1 = app.invoke(
    {"messages": [{"role": "user", "content": "i'd like to speak to Bob"}]},
    config,
)
print("----------------------------- RESULT FOR TURN 1 -----------------------------")
for message in turn_1['messages']:
    if isinstance(message, HumanMessage):
        print("[Human Message]")
    elif isinstance(message, AIMessage):
        print("[AI Message]")
    elif isinstance(message, ToolMessage):
        print("[Tool Message]")
    else:
        print("[Unknown Message Type]")
    
    print(message.content)
    print("-" * 40)
turn_2 = app.invoke(
    {"messages": [{"role": "user", "content": "what's 5 + 7?"}]},
    config,
)
print("----------------------------- RESULT FOR TURN 2 -----------------------------")
for message in turn_2['messages']:
    if isinstance(message, HumanMessage):
        print("[Human Message]")
    elif isinstance(message, AIMessage):
        print("[AI Message]")
    elif isinstance(message, ToolMessage):
        print("[Tool Message]")
    else:
        print("[Unknown Message Type]")
    
    print(message.content)
    print("-" * 40)

""" 
This code will create a graph of above code and save as graph_output.png in local folder
"""
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = app.get_graph().draw_mermaid_png()

# Save it to a file
with open("swarm_api.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("swarm_api.png")
img.show()



# response = swarm.invoke({
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         ]
#     })  
# for message in response['messages']:
#     if isinstance(message, HumanMessage):
#         print("[Human Message]")
#     elif isinstance(message, AIMessage):
#         print("[AI Message]")
#     elif isinstance(message, ToolMessage):
#         print("[Tool Message]")
#     else:
#         print("[Unknown Message Type]")
    
#     print(message.content)
#     print("-" * 40)