from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint
from typing import Optional, Any

@entrypoint(checkpointer=MemorySaver())
def my_workflow(input_data: str, previous: Optional[str] = None) -> str:
    if previous:
        return previous + " " + "world"
    return input_data

config = {
    "configurable": {
        "thread_id": "some_thread"
    }
}
#You want the return value to be remembered next time → previous only
print(my_workflow.invoke("hello",config=config))
print(my_workflow.invoke("call",config=config))

@entrypoint(checkpointer=MemorySaver())
def logic(x: int, *, previous: int | None = None) -> entrypoint.final[str, int]:
    previous = previous or 0
    new_state = previous + x
    return entrypoint.final(
        value=f"User sees: {previous} + {x} = {new_state}",
        save=new_state
    )
#You want to return something different than what gets remembered
print(logic.invoke(4,config))
print(logic.invoke(6,config))

"""

entrypoint.final
entrypoint.final is a special primitive that can be returned from an entrypoint and allows decoupling the value that is saved in the checkpoint from the return value of the entrypoint.

The first value is the return value of the entrypoint, and the second value is the value that will be saved in the checkpoint. The type annotation is entrypoint.final[return_type, save_type].

"""