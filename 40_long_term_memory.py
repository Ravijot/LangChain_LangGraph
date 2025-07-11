from langchain_core.runnables import RunnableConfig
from langgraph.store.memory import InMemoryStore
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import Optional
from langchain.embeddings import init_embeddings
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)



llm = init_chat_model("openai:gpt-4o-mini")

# Create store with semantic search enabled
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

def chat(state, *, store: BaseStore):
    # Search based on user's last message
    items = store.search(
        ("user_123", "memories"), query=state["messages"][-1].content, limit=2
    )
    # print(items)
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""
    response = llm.invoke(
        [
            {"role": "system", "content": f"You are a helpful assistant.\n{memories}"},
            *state["messages"],
        ]
    )
    # print(response)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

response = graph.invoke({"messages": [{"role": "user", "content": "I'm hungry"}]})


for message in response['messages']:
    if isinstance(message, HumanMessage):
        print("[Human Message]")
    elif isinstance(message, AIMessage):
        print("[AI Message]")
    elif isinstance(message, ToolMessage):
        print("[Tool Message]")
    else:
        print("[Unknown Message Type]")
    
    print(message.content)
    print("-" * 40)  # optional separator