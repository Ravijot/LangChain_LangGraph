from typing import Annotated, TypedDict, List
from langchain_core.tools import InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.types import Command
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

# Define a custom state to hold messages + user name
class CustomState(AgentState):
    user_name: str  # <- This is our short-term memory slot

# Tool 1: Simulates looking up user info
def user_info(
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig
) -> Command:
    """Looks up user information and updates state with user name."""
    user_id = config["configurable"].get("user_id")
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        "messages": [
            ToolMessage(
                content="User information successfully retrieved.",
                tool_call_id=tool_call_id
            )
        ]
    })

# Tool 2: Greets the user using short-term memory
def greet(
    state: Annotated[CustomState, InjectedState]
) -> str:
    """Greets the user with their name."""
    user_name = state.get("user_name", "Guest")
    return f"Hello {user_name}! Welcome back."



# Create ReAct agent with tools and state memory
agent = create_react_agent(
    model=model,
    tools=[user_info, greet],
    state_schema=CustomState
)

# Run the agent with a user prompt and user ID
result = agent.invoke(
    {"messages": [HumanMessage(content="greet the user")]},
    config={"configurable": {"user_id": "user_123"}}
)

# Print final response
print(result["messages"][-1].content)
print(result)


