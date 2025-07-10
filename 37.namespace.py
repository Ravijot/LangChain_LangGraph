"""

LangGraph stores long-term memories as JSON documents in a store. Each memory is organized under a custom namespace (similar to a folder) 
and a distinct key (like a file name). Namespaces often include user or org IDs or other labels that makes it easier to organize information.
This structure enables hierarchical organization of memories. Cross-namespace searching is then supported through content filters.

"""
from langgraph.store.memory import InMemoryStore
import uuid

in_memory_store = InMemoryStore()

"""
Memories are namespaced by a tuple, which in this specific example will be (<user_id>, "memories").
The namespace can be any length and represent anything, does not have to be user specific.
"""
user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
user_id ="2"
namespace_for_memory = (user_id,"memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference":"I love burger"}
in_memory_store.put(namespace_for_memory, memory_id, memory)
"""
We use the store.put method to save memories to our namespace in the store. When we do this, we specify the namespace,
as defined above, and a key-value pair for the memory: the key is simply a unique identifier for the memory (memory_id) 
and the value (a dictionary) is the memory itself."""

memories = in_memory_store.search(namespace_for_memory)
print("All memories : ",memories)
print("Latest Memory : ",memories[-1].dict())


