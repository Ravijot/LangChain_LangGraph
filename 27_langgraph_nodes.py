from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_core.runnables import RunnableConfig

"""
In LangGraph, nodes are typically python functions (sync or async) where the first positional argument is the state, and (optionally), the second positional argument is a "config", containing optional configurable parameters (such as a thread_id).
When creating a graph, you can also mark that certain parts of the graph are configurable. This is commonly done to enable easily switching between models or system prompts.
"""

# 1. Define the state
class State(TypedDict):
    input: str
    output: str

# 2. Create a node that uses `RunnableConfig`
def greet_node(state: State, config: RunnableConfig) -> dict:
    user_id = config.get("configurable", {}).get("user_id", "guest")
    style = config.get("configurable", {}).get("greeting_style", "casual")
    
    if style == "formal":
        greeting = f"Good day to you, {user_id}."
    else:
        greeting = f"Hey {user_id}!"

    return {"output": greeting}

# 3. Build the LangGraph
builder = StateGraph(State)
builder.add_node("greet_node", greet_node)
builder.set_entry_point("greet_node")
builder.set_finish_point("greet_node")
graph = builder.compile()

# 4. Invoke it with config
output = graph.invoke(
    {"input": "hello"},
    config={
        "configurable": {
            "user_id": "John Doe",
            "greeting_style": "formal"
        }
    }
)

print(output)