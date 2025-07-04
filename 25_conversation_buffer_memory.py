from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

# Step 1: Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Step 2: Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # pulls in memory
    ("human", "{input}")
])

# Step 3: Set up LLM and LCEL chain
chain =  prompt | model

# Inputs list
inputs = [
    "Hi, I am Alice",
    "Do you know my name?",
    "Thanks, bye!"
]

# Traditional for loop
for user_input in inputs:
    memory_vars = memory.load_memory_variables({})
    print("Memory Vars : ",memory_vars)
    full_input = {"input": user_input, **memory_vars}
    response = chain.invoke(full_input)

    print(f"\n🧑: {user_input}")
    print(f"🤖: {response.content}")

    memory.save_context({"input": user_input}, {"output": response.content})
    print("Memory : ",memory)