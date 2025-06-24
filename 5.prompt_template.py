import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage


try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
prompt_template1 = PromptTemplate.from_template("Tell me a {tone} joke about {topic} for a {audience}.")
formatted_prompt = prompt_template1.format(
    tone="sarcastic",
    topic="AI",
    audience="developers"
)

print(formatted_prompt)
model = init_chat_model("gpt-4o-mini", model_provider="openai")
# print(model.invoke(formatted_prompt).content)

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me a joke about {topic}")
])

messages = prompt_template.format(topic="cat")
print(messages)
print(model.invoke(messages).content)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="msgs")
])


chat_history = [HumanMessage(content="hi!")]
messages = prompt_template.format_messages(msgs=chat_history)
print(messages)
response = model.invoke(messages)
print(response.content)
