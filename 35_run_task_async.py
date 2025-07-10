import asyncio
import random
from typing import Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint

# Simulated slow async computation
async def slow_computation(x: int) -> int:
    print(f"⏳ Starting computation for {x}")
    await asyncio.sleep(1 + random.random())  # Simulate variable delay
    return x * 10

# Memory-based checkpointer
checkpointer = MemorySaver()

# Async entrypoint
@entrypoint(checkpointer=checkpointer)
async def my_workflow(some_input: int, previous: Optional[int] = None) -> int:
    print(f"🚀 [my_workflow] thread={some_input}, previous={previous}, input={some_input}")
    result = await slow_computation(some_input)
    print(f"✅ Done for input {some_input}: result={result}")
    return result

# Run multiple workflows concurrently with optional delay between launches
async def run_multiple_async_workflows():
    async def run_with_delay(i: int):
        await asyncio.sleep(i * 0.5)  # Delay between starting each task
        config = {
            "configurable": {
                "thread_id": f"async-thread-{i}"
            }
        }
        result = await my_workflow.ainvoke(i, config=config)
        print(f"🧵 Thread {i}: result={result}")
        return result

    # Launch 5 workflows with staggered starts
    tasks = [run_with_delay(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print("\n🎉 All results:", results)

# Run it
if __name__ == "__main__":
    asyncio.run(run_multiple_async_workflows())
