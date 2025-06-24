import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import  BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
import requests

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["weather_api_key"] = os.getenv("weather_api_key")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")

model = init_chat_model("gpt-4o-mini", model_provider="openai")
api_key = os.getenv('weather_api_key')

class Input(BaseModel):
    city: str = Field(description="Name of the city which data need to be retrieved")
    
class Tools:
    
    def __init__(self):
        self.tools = []
        self.add_tool()
        
    def get_weather(self,city):
        print("in current weather")
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric"  # Optional, for temperature in Celsius
        }
        response = requests.get(base_url, params=params)
        print(response.text)
        return response.json()

    def add_tool(self):
        weather= StructuredTool.from_function(
            func=self.get_weather,
            name="Weather",
            description="Useful in getting current weather",
            args_schema=Input,
            return_direct=False,
           
        )
        self.tools.append(weather)
        
    def get_tools(self):
        return self.tools
    
toolkit = Tools().get_tools()
prompt = PromptTemplate.from_template("You are a helpful assistant that can answer questions about the weather. Use the tools provided to get the current weather information about user query. \n\n {input}\n\n{agent_scratchpad} \n\n {chat_history}")
# Create the agent
agent = create_tool_calling_agent(llm=model, tools=toolkit, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True, return_intermediate_steps=True)

response = agent_executor.invoke({"input": "What is the current weather in Delhi?", "chat_history": []})
print(response)
print(response['intermediate_steps'])