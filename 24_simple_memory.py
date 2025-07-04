import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.memory import SimpleMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)

# Step 1: Define static memory
memory = SimpleMemory(memories={"user_name": "Alice", "user_location": "Paris"})

# Step 2: Create a prompt with user name and static context
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant talking to {user_name} from {user_location}."),
    ("human", "{input}")
])

# Step 3: Set up the LLM and chain
chain = prompt | model

# Step 4: Hardcoded conversation steps (scripted)
conversation = [
    {"input": "Hi!", "expected_output": "Hello Alice! How can I help you today?"},
    {"input": "Where am I?", "expected_output": "You're in Paris."},
    {"input": "What’s the weather like?", "expected_output": "I don’t have real-time data, but Paris usually has mild weather."},
    {"input": "Bye!", "expected_output": "Goodbye Alice!"}
]

# Step 5: Run through the conversation
for step in conversation:
    inputs = {"input": step["input"], **memory.load_memory_variables({})}
    response = chain.invoke(inputs)
    print(f"\n🧑: {step['input']}")
    print(f"🤖 (LLM): {response.content}")
    print(f"✅ Expected: {step['expected_output']}")

