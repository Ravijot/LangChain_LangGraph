from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.func import entrypoint
from typing import TypedDict, Literal, Union, Annotated
import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

# Define the state
class GraphState(TypedDict):
    tool_selected: Union[str, None]
    input_value1: int
    input_value2: int
    result: Union[int, None]
    human_approval: Union[Literal["approved"], Literal["rejected"], None]
    human_input : str

# Tools@tool

def multiply(state: Annotated[GraphState, "state"]) -> GraphState:
    """Multiply two numbers."""
    state["result"]=state["input_value1"]*state["input_value2"]
    return state


def divide(state: Annotated[GraphState, "state"]) -> GraphState:
    """Divide two numbers"""
    state["result"] = state["input_value1"] // state["input_value2"] if state["input_value2"] != 0 else 0
    return state

# Tool Selector

def tool_selector(state: Annotated[GraphState, "state"]) -> GraphState:
    print("Selecting tool...")
    if state["input_value1"]<10 and state['input_value1'] < 10:
        print("Multiply Tool Selected")
        
        state["tool_selected"]="multiply"
        return state
    else:
        print("Divide Tool Selected")
        state["tool_selected"]="divide"
        return state

# Human Review

def human_review(state: Annotated[GraphState, "state"]) -> Command[Literal["multiply", "divide"]]:
    print(f"\n[Human Review] Inputs: {state['input_value1']} and {state['input_value2']}, Tool: {state['tool_selected']}")
    
    decision = interrupt({"task": "Review the output from the LLM and make any necessary edits.",
                          "tool_selected": state["tool_selected"]
           })
    print("Resume")
    print("Decision:", decision)
    
    if decision['human_input'] == "a":
        print("Approved by Human")
        state["human_approval"]="approved"

        return Command(goto="multiply", update={"tool_selected": "multiply", "human_approval": "approved","human_input": decision['human_input']}, resume=True)

    else:
        print("Rejected by Human")
        state["human_approval"]= "rejected"
        return Command(goto="divide", update={"tool_selected": "divide", "human_approval": "rejected", "human_input": decision['human_input']}, resume=True)

# Run Tool

def run_tool(state: Annotated[GraphState, "state"]) -> GraphState:
    print("Going to Run Tool")
    if state["tool_selected"] == "multiply":
        print("Executing Multiply Tool")
        result = multiply.invoke({"a":state["input_value1"],"b": state["input_value2"]})
    elif state["tool_selected"] == "divide":
        print("Executing Divide Tool")
        result = divide.invoke({"a" : state["input_value1"], "b" :state["input_value2"]})
    else:
        result = None
    state["result"]=result
    return state

# Build the graph
builder = StateGraph(GraphState)
builder.add_node("tool_selector", tool_selector)
builder.add_node("human_approval", human_review)
builder.add_node("multiply", multiply)
builder.add_node("divide", divide)

builder.set_entry_point("tool_selector")
builder.add_edge("tool_selector", "human_approval")
builder.add_edge("multiply", END)
builder.add_edge("divide", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Run until interrupt
config = {"configurable": {"thread_id": uuid.uuid4()}}
result1 = graph.invoke({'input_value1':8,'input_value2':6}, config=config)
print(result1)
print(result1['__interrupt__'])
human_review = input("Please provide your input (a/r): ").strip().lower()
result2 = graph.invoke(Command(resume={"human_input": human_review}), config=config)
print("Final Result after Human Input:")
print(result2)
