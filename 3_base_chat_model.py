from email.mime import message
import os
from xml.parsers.expat import model 
from dotenv import load_dotenv
from typing import List, Optional, Iterator
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult
)   

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
def call_the_model(message):
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)
    response = model.invoke(message)
    return response

def call_the_model_streaming(message):
    model = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.8)
    return model.stream(message)

class CustomChatModel(BaseChatModel):
    
    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "my-custom-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
    ) -> ChatResult:
        # Simple logic to echo user message with timestamp
        user_input = messages[-1].content if messages else "Hello"
        response_text = call_the_model(user_input)

        ai_message = AIMessage(content=response_text.content)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
    ) -> Iterator[ChatGenerationChunk]:
        user_input = messages[-1].content if messages else "Hello"
        for ch in call_the_model_streaming(user_input):
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=ch.content)
            )

    def get_tokenizer(self):
        # Dummy tokenizer using whitespace
        return lambda text: text.split()

    def get_num_tokens(self, text: str) -> int:
        tokenizer = self.get_tokenizer()
        return len(tokenizer(text))
    
model = CustomChatModel()

# Normal invoke
response = model.invoke([HumanMessage(content="What's up?")])
print("Response:", response.content)

# Streaming (simulate chunked output)
for chunk in call_the_model_streaming("tell me a joke about AI."):
    print(chunk.content, end="", flush=True)