from langchain_openai import OpenAIEmbeddings
import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
documents = [
    "Common side effects of taking ibuprofen daily include stomach irritation, ulcers, kidney problems, and increased risk of heart issues. It's advised not to exceed 800mg per dose or 3200mg per day.",
    "Supervised learning uses labeled datasets to train models — examples include classification and regression. Unsupervised learning finds patterns in unlabeled data, like clustering or dimensionality reduction.",
    "Slow internet during video calls is often due to high network usage or weak Wi-Fi signal. Solutions include using a wired connection, closing background apps, or upgrading your router.",
    "Amazon's refund policy allows returns within 30 days of purchase for most items. Refunds are issued to the original payment method once the item is received."
]


vector_store = Chroma.from_texts(
    texts=documents,
    collection_name="example_collection",
    embedding=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

retriever = vector_store.as_retriever()
docs = retriever.invoke("Why does my stomach hurt after taking ibuprofen?")
print(docs)