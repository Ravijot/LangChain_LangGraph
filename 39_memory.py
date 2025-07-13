"""
| **Feature**     | **Short-Term Memory (STM)**                                            | **Long-Term Memory (LTM)**                                   |
| --------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Scope**       | Limited to a single graph run or thread execution                      | Persistent across many runs, threads, or user sessions       |
| **Stored In**   | `State` object during graph execution (possibly passed via `previous`) | External storage (e.g., vector store, Redis, database, etc.) |
| **Example Use** | Track current messages or tool outputs                                 | Recall past conversations, FAQs, user preferences            |
| **Lifespan**    | Ephemeral – exists only during current execution                       | Persistent – survives restarts and multiple sessions         |
| **Use Case**    | Maintain local context for one interaction                             | Retrieve historical or contextual information over time      |
| **Where Used**  | In node logic or entrypoint via state and transitions                  | In memory-backed retrievers (LangChain, custom store, etc.)  |

| **Type**       | **Description**                                            | **Example**                                                 |
| -------------- | ---------------------------------------------------------- | ----------------------------------------------------------- |
| **Short-Term** | In-memory state used during graph execution; not persisted | Tool results, input/output messages in one graph run        |
| **Long-Term**  | Persisted external memory used across sessions             | User profile stored in Redis or vector store (e.g., Chroma) |

| **Concept**      | **Role**                                                                 |
| ---------------- | ------------------------------------------------------------------------ |
| **Checkpointer** | Used to **persist STM state** between runs (for resuming or rollback)    |
| **Store**        | External backend used to persist **LTM**, e.g., Redis, Chroma, Weaviate  |
| **Namespace**    | Logical partitioning of LTM per user/task in the store (isolation layer) |

"""