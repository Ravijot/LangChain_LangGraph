from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
import uuid

@task
def add_one(number: int) -> int:
    return number + 1

checkpointer = MemorySaver()

@entrypoint(checkpointer=checkpointer)
def graph(numbers: list[int]) -> list[str]:
    futures = [add_one(i) for i in numbers]
    return [f.result() for f in futures]

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}

# print(graph.invoke([1,2,3,4],config))
for item in graph.invoke([1,2,3,4],config):
    print(item)