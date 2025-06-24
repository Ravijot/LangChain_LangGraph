import os
from dotenv import load_dotenv
import requests
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
try:
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["weather_api_key"] = os.getenv("weather_api_key")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")

# OpenAI and Weather API keys
api_key = os.getenv('weather_api_key')

# Initialize ChatOpenAI model with streaming
stream_handler = StreamingStdOutCallbackHandler()
model = ChatOpenAI(model="gpt-4o", streaming=True, callbacks=[stream_handler])

# Pydantic schema for tool input
class Input(BaseModel):
    city: str = Field(description="Name of the city which data need to be retrieved")

# Tool class with weather fetching
class Tools:
    def __init__(self):
        self.tools = []
        self.add_tool()

    def get_weather(self, city):
        print("Calling OpenWeatherMap API...")
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"
        }
        response = requests.get(base_url, params=params)
        print(f"API Response: {response.text}")
        return response.json()

    def add_tool(self):
        weather_tool = StructuredTool.from_function(
            func=self.get_weather,
            name="Weather",
            description="Useful in getting current weather",
            args_schema=Input,
            return_direct=False,
        )
        self.tools.append(weather_tool)

    def get_tools(self):
        return self.tools

# Setup tools
toolkit = Tools().get_tools()

# Prompt template
prompt = PromptTemplate.from_template(
    "You are a helpful assistant that can answer questions about the weather. "
    "Use the tools provided to get the current weather information about user query.\n\n"
    "{input}\n\n{agent_scratchpad} \n\n{chat_history}"
)

# Create agent and executor
agent = create_tool_calling_agent(llm=model, tools=toolkit, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

# Stream response from agent
print("Asking the agent...")
for chunk in agent_executor.stream({
    "input": "What is the current weather in Delhi?",
    "chat_history": []
}):
    pass  # Output is already printed by StreamingStdOutCallbackHandler