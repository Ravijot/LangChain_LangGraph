from langchain_core.runnables import RunnableLambda

#a RunnableLambda is a special wrapper that converts any Python callable (like a def or lambda function) into a Runnable—LangChain's universal interface for building pipelines.
def word_count(text: str) -> int:
    return len(text.split())

wc = RunnableLambda(word_count)
print(wc.invoke("hello world from langchain"))  # ➝ 4
print(wc.batch(["a b", "one two three"]))       # ➝ [2, 3]
