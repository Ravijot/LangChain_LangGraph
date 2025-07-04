import time
from langgraph.cache.memory import InMemoryCache
from langgraph.func import entrypoint, task
from langgraph.types import CachePolicy


@task(cache_policy=CachePolicy(ttl=120))  
def slow_add(x: int) -> int:
    time.sleep(1)
    return x * 2


@entrypoint(cache=InMemoryCache())
def main(inputs: dict) -> dict[str, int]:
    result1 = slow_add(inputs["x"]).result()
    result2 = slow_add(inputs["x"]).result()
    return {"result1": result1, "result2": result2}


for chunk in main.stream({"x": 5}, stream_mode="updates"):
    print(chunk)

