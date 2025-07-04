
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

checkpointer = MemorySaver()
@task
def generate_paragraph(topic: str) -> str:
    """A simple workflow that writes an essay and asks for a review."""
    essay = model.invoke([
        {"role": "system", "content": "You are a helpful assistant that writes essays."},
        {"role": "user", "content": f"Write an short paragraph of 4 or 5 lines {topic}."}
    ]).content

    return essay
    
thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


@entrypoint(checkpointer=checkpointer)
def workflow(topics: list[str]) -> str:
    """Generates multiple paragraphs in parallel and combines them."""
    futures = [generate_paragraph(topic) for topic in topics]
    paragraphs = [f.result() for f in futures]
    return "\n\n".join(paragraphs)

# Run the workflow
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
# for item in  workflow.invoke(["quantum computing", "climate change"], config=config):
#     print(item)
results = workflow.invoke(["quantum computing", "climate change"], config=config)
print(results)