import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

chain = (
    PromptTemplate.from_template(
        """Given the user question below, classify it as either being about `Physics`, `Computer Science`, or `Other`.

Do not respond with more than one word.

<question>
{question}
</question>

Classification:"""
    )
    | llm
    | StrOutputParser()
)

first_prompt = PromptTemplate.from_template(
    "You a Physics Professor help student to solve thier query. {question}"
)
chain_one = first_prompt | llm | StrOutputParser()

second_prompt = PromptTemplate.from_template(
    "You are a Senior Software Engineer help computer science students to solve thier queries.{question}"
)
chain_two = second_prompt | llm | StrOutputParser()

second_third = PromptTemplate.from_template(
    "You are helpful assistant{question}"
)
chain_three = second_third | llm | StrOutputParser()

def route(info):
    print(info['topic'])
    if "Physics".lower() in info["topic"].lower():
        print("Routing to Physics chain")
        return chain_one
    elif "Computer Science".lower() in info["topic"].lower():
        print("Routing to Computer Science chain")
        return chain_two
    else:
        print("Routing to Other chain")
        return chain_three
    
# Topics will return a topic "Physics", "Computer Science" or "Other", then second statement extract question from the input, then it pass to route function.
full_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(
    route
)

print(full_chain.invoke({"question": "Give me a Cpp fibonacci series program?"}))

#One can also refer to RunnableBranch to create a more complex routing logic.