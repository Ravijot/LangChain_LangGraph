from typing import TypedDict
from langgraph.graph import StateGraph, END, START

#Input from User
class InputState(TypedDict):
    user_input : str
    
#Final Output
class OutputState(TypedDict):
    graph_output : str

class OverallState(TypedDict):
    raw_text : str
    issue_summary : str
    graph_output : str
    issue_type: str
    
    
# Node 1: Clean and store user input
def preprocess_input(state: InputState) -> OverallState:
    text = state["user_input"].strip().lower()
    return {"raw_text": text}

# Node 2: Classify issue based on simple keywords (simulating LLM/tool)
def classify_issue(state: OverallState) -> OverallState:
    text = state["raw_text"]
    if "internet" in text or "wifi" in text:
        issue = "Internet Issue"
    elif "bill" in text or "payment" in text:
        issue = "Billing Issue"
    elif "slow" in text:
        issue = "Speed Issue"
    else:
        issue = "General Inquiry"
    return {"issue_type": issue}

# Node 3: Generate a formatted ticket
def generate_ticket(state: OverallState ) -> OutputState:
    return {
        "graph_output": f"Ticket Created: {state['issue_type']} | Details: {state['raw_text'].capitalize()}"
    }
    
# ------------------ Build the graph ------------------ #
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

builder.add_node("preprocess_input", preprocess_input)
builder.add_node("classify_issue", classify_issue)
builder.add_node("generate_ticket", generate_ticket)

builder.add_edge(START, "preprocess_input")
builder.add_edge("preprocess_input", "classify_issue")
builder.add_edge("classify_issue", "generate_ticket")
builder.add_edge("generate_ticket", END)

graph = builder.compile()

# ------------------ Test it ------------------ #
output = graph.invoke({"user_input": "My internet has been down for hours!"})
print(output)
""" 
This code will create a graph of above code and save as graph_output.png in local folder
"""
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = graph.get_graph().draw_mermaid_png()

# Save it to a file
with open("graph_output.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("graph_output.png")
img.show()