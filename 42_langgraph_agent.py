import os 
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import  BaseModel, Field
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver


try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

api_key = os.getenv('weather_api_key')

class WeatherResponse(BaseModel):
    temperature : str = Field(description="Current temperature in the city")
    weather_details : str = Field(description="Give all weather details in the city")


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
    

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=Tools().get_tools(),
    checkpointer=checkpointer,
    prompt="You are a helpful weather assistant" ,
    response_format=WeatherResponse
)

# Run the agent
config = {"configurable": {"thread_id": "1"}}

agent_response = agent.invoke(
    {"messages": [{"role": "user", "content": "what about new york?"}]},
    config
)
# print(agent_response['messages'][-1].content)
print(agent_response["structured_response"])