import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    

# Your own wrapper for model init
def initialize_model(model_name, model_provider="openai", temperature=0.8):
    model = init_chat_model(
        model=model_name,
        model_provider=model_provider,
        temperature=temperature
    )
    return model

class CustomLLM(LLM):
    """Custom LLM wrapper using OpenAI gpt-4o-mini for both invoke and stream."""

    model_name: str = "gpt-4o-mini"
    temperature: float = 0.8
    model_provider: str = "openai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        model = initialize_model(self.model_name, temperature=self.temperature)
        response = model.invoke(prompt)
        return response.content  # Extract string from AIMessage

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        model = initialize_model(self.model_name, temperature=self.temperature)
        stream = model.stream(prompt)  # Returns AIMessageChunk stream

        for chunk in stream:
            text = chunk.content or ""
            gen_chunk = GenerationChunk(text=text)

            if run_manager:
                run_manager.on_llm_new_token(text, chunk=gen_chunk)

            yield gen_chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
        }

    @property
    def _llm_type(self) -> str:
        return "custom_openai_wrapper"

llm = CustomLLM()

print(llm._llm_type)
print(llm._identifying_params)
# print(llm._call("Hello, how are you?"))
for chunk in llm.stream("how to start a car"):
    print(chunk, end="")