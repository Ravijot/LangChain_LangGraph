from langchain import hub
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent, AgentOutputParser
from langchain_core.prompts import PromptTemplate
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import StructuredTool, tool
from pydantic import  BaseModel, Field
import requests

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)
api_key = os.getenv('weather_api_key')

class Input(BaseModel):
    city: str = Field(description="Name of the city which data need to be retrieved")
    
class Tools:
    
    def __init__(self):
        self.tools = []
        self.add_tool()
        
    def get_weather(self,city):
        
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

tools = Tools().get_tools()

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,verbose=True)

print(agent_executor.invoke({"input":"Give weather details of Delhi?"}))
