import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnableLambda

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai")
parser = StrOutputParser()

sentiment_prompt = PromptTemplate.from_template(
    "Classify the sentiment (positive, negative, neutral) of this message:\n\n{input}"
)
sentiment_chain = sentiment_prompt | model | parser

response_prompt = PromptTemplate.from_template(
    "Given the sentiment '{sentiment}', write a short empathetic reply to the user."
)

chain = sentiment_chain | \
        RunnableLambda(lambda sentiment: {"sentiment": sentiment}) | \
        response_prompt | \
        model | \
        parser

user_message = "I failed my interview and feel like I’ll never get a job."
result = chain.invoke({"input": user_message})
print(result)
