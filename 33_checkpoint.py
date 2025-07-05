"""
What Is a Checkpointer?
A checkpointer is used to:
Persist the intermediate states of a LangGraph flow.
Resume execution from the last saved point (in case of failure, interruption, or async execution).
Enable replayability, debugging, and observability of a graph's execution over time.

It’s especially useful in long-running or multi-agent applications, where you may want to:
Pause and resume execution.
Review how the system reached a certain state.
Handle retries or errors gracefully.

Example Checkpointers
LangGraph supports different checkpointing backends:

Checkpointer Type	Description
MemorySaver	In-memory (for testing or short-lived graphs).
SQLiteSaver	Stores checkpoints in a local SQLite database.
MongoDBSaver	Stores checkpoints in a MongoDB database (for scaling).
LangSmithSaver	Integrates with LangSmith for advanced observability.
Custom	You can implement your own checkpointing logic.

Is the checkpointer just memory?
Not exactly. The checkpointer is an interface (a pattern) that can save the state of the graph — where and how it saves depends on which type of checkpointer you use.

MemorySaver:
Stores checkpoints in memory (RAM).
Temporary — once your Python program ends, all checkpoints are gone.
Good for: testing, learning, debugging short-lived flows.

Checkpoints
The state of a thread at a particular point in time is called a checkpoint. Checkpoint is a snapshot of the graph state saved at each super-step and is represented by StateSnapshot object with the following key properties:

config: Config associated with this checkpoint.
metadata: Metadata associated with this checkpoint.
values: Values of the state channels at this point in time.
next A tuple of the node names to execute next in the graph.
tasks: A tuple of PregelTask objects that contain information about next tasks to be executed. If the step was previously attempted, it will include error information. If a graph was interrupted dynamically from within a node, tasks will contain additional data associated with interrupts.
Checkpoints are persisted and can be used to restore the state of a thread at a later time.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": ""}, config))

# get the latest state snapshot

config = {"configurable": {"thread_id": "1"}}
print(graph.get_state(config))

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
print(graph.get_state(config))

config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
"""
Use Cases
Use Case	                Thread ID	                                    Checkpoint ID
Run a new session	        Set a unique thread_id like "run_2025_07_04"	Not needed (auto-generated)
Resume a session	        Use the same thread_id	                        Omit checkpoint_id (picks up latest)
Roll back to a past state	Use the same thread_id	                        Provide specific checkpoint_id
Debug or inspect	        Use get_state()	                                Provide both IDs
"""