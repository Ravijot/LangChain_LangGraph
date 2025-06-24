import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai")


class ResponseFormatter(BaseModel):
    emotion : str = Field(description="The emotion conveyed in the response e.g. happy, sad, angry, etc.")
    confidence_score : float = Field(description="The confidence score of the emotion conveyed in the response, between 0 and 1.")

prompt = PromptTemplate(
    input_variables=["input"],
    template="What emotion is conveyed in the following text? {input}",
)
formatted_prompt = prompt.format(input="Hurrah! I just won the lottery!")
chain = model | StrOutputParser()
response = chain.invoke(formatted_prompt)
print(response)

parser = PydanticOutputParser(pydantic_object=ResponseFormatter)

prompt = PromptTemplate(
    input_variables=["text"],
    template="What emotion is conveyed in the following text?\n\n{text}\n\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser
response = chain.invoke({"text": "Hurrah! I just won the lottery!"})
print(response)
