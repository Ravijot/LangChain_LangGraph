"""
The Functional API uses two key building blocks:

@entrypoint – Marks a function as the starting point of a workflow, encapsulating logic and managing execution flow, including handling long-running tasks and interrupts.

@task – Represents a discrete unit of work, such as an API call or data processing step, that can be executed asynchronously within an entrypoint. Tasks return a future-like object that can be awaited or resolved synchronously.

Below we demonstrate a simple application that writes an essay and interrupts to request human review.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.types import interrupt
import time
import uuid
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

@task
def write_essay(topic: str) -> str:
    """Write an essay about the given topic."""
    time.sleep(1) # A placeholder for a long-running task.
    return f"An essay about topic: {topic}"

"""The @entrypoint decorator can be used to create a workflow from a function. It encapsulates workflow logic and manages execution flow, including handling long-running tasks and interrupts.
An entrypoint is defined by decorating a function with the @entrypoint decorator.

The function must accept a single positional argument, which serves as the workflow input. If you need to pass multiple pieces of data, use a dictionary as the input type for the first argument.

Decorating a function with an entrypoint produces a Pregel instance which helps to manage the execution of the workflow (e.g., handles streaming, resumption, and checkpointing)."""

@entrypoint(checkpointer=MemorySaver())
def workflow(topic: str) -> dict:
    """A simple workflow that writes an essay and asks for a review."""
    essay = model.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes essays."},
        {"role": "user", "content": f"Write an short paragraph of 4 or 5 lines {topic}."}
    ]).content
    is_approved = interrupt({
        # Any json-serializable payload provided to interrupt as argument.
        # It will be surfaced on the client side as an Interrupt when streaming data
        # from the workflow.
        "essay": essay, # The essay we want reviewed.
        # We can add any additional information that we need.
        # For example, introduce a key called "action" with some instructions.
        "action": "Please approve/reject the essay",
    })

    return {
        "essay": essay, # The essay that was generated
        "is_approved": is_approved, # Response from HIL
    }
    
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

for item in workflow.stream("Morning Walk", config):
    print(item)

from langgraph.types import Command

# Get review from a user (e.g., via a UI)
# In this case, we're using a bool, but this can be any json-serializable value.
human_review = True

for item in workflow.stream(Command(resume=human_review), config):
    print(item)