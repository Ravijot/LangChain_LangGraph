from langgraph.store.memory import InMemoryStore
import uuid
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)




user_id = "1"
namespace_for_memory = (user_id, "memories")
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)
memory_id = str(uuid.uuid4())
memory = {"food_preference":"I dont like Chinese food"}
store.put(namespace_for_memory, memory_id, memory)

memories = store.search(
    namespace_for_memory,
    query="Which food user dont like?",
    limit=3  # Return top 3 matches
)
print("Retrieved memory : ",memories)



