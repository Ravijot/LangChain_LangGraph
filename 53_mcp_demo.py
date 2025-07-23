import os
import asyncio
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

async def main():
    # Load environment variables
    try:
        load_dotenv()
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        print("Environment variables loaded successfully.")
    except Exception as e:
        print(f"Error loading environment variables: {e}")

    # Initialize model
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

    # Initialize MCP client with tool definitions
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["51_mcp_math.py"],  # Use absolute path if needed
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            }
        }
    )

    # Await tools
    tools = await client.get_tools()

    # Create ReAct agent
    agent = create_react_agent(model, tools)

    # Invoke agent for math
    # math_response = await agent.ainvoke(
    #     {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
    # )
    # print("Math response")
    # for message in math_response['messages']:
    #     if isinstance(message, HumanMessage):
    #         print("[Human Message]")
    #     elif isinstance(message, AIMessage):
    #         print("[AI Message]")
    #     elif isinstance(message, ToolMessage):
    #         print("[Tool Message]")
    #     else:
    #         print("[Unknown Message Type]")
        
    #     print(message.content)
    #     print("-" * 40)

    # Invoke agent for weather
    weather_response = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "what is the weather in Delhi?"}]}
    )
    print("Weather response")
    for message in weather_response['messages']:
        if isinstance(message, HumanMessage):
            print("[Human Message]")
        elif isinstance(message, AIMessage):
            print("[AI Message]")
        elif isinstance(message, ToolMessage):
            print("[Tool Message]")
        else:
            print("[Unknown Message Type]")
        
        print(message.content)
        print("-" * 40)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
