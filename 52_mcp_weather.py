from mcp.server.fastmcp import FastMCP
import requests
import os

from dotenv import load_dotenv
load_dotenv()
mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get weather for location."""
    api_key = os.getenv('weather_api_key')
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
            "q": city,  
            "appid": api_key,
            "units": "metric"  # Optional, for temperature in Celsius
        }
    response = requests.get(base_url, params=params)
    data = response.json()
    weather = f"Weather Details : {data}"
    return weather
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
    
#Before running this file run the 52_mcp_weather.py file to start the MCP server with command : python 52_mcp_weather.py
#We are going to use this file as a tool for math operations in the MCP server in 53_mcp_demo.py