import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai")
parser = StrOutputParser()

prompt1 = PromptTemplate.from_template("What is the capital of {place}?")
prompt2 = PromptTemplate.from_template("What is the population of {place}?")

# Build two chains
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser


parallel_chain = RunnableParallel({
    "capital": chain1,
    "population": chain2
})

# Run it
result = parallel_chain.invoke({"place": "Japan"})
print(result)
