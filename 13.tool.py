from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
   """Multiply two numbers."""
   return a * b

multiply.invoke({"a": 2, "b": 3})

print(multiply.name) # multiply
print(multiply.description) # Multiply two numbers.
print(multiply.args) # {'a': 2, 'b': 3}

print(multiply.invoke({"a": 2, "b": 3})) # 6