import os 
import uuid
from base64 import b64decode
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import UnstructuredPDFLoader
from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from unstructured.chunking.basic import chunk_elements
from unstructured.chunking.title import chunk_by_title
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage

try: 
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    print("Environment variables loaded successfully.")
except Exception as e:
    print(f"Error loading environment variables: {e}")
    
model = init_chat_model("gpt-4o", model_provider="openai", temperature=0.8)

# Extract images, tables
elements = partition_pdf(
    filename="wildlife.pdf",                  # mandatory
    strategy="hi_res",                                     # mandatory to use ``hi_res`` strategy
    extract_images_in_pdf=True,                            # mandatory to set as ``True``
    extract_image_block_types=["Image", "Table"],          # optional
    extract_image_block_to_payload=True,    
    infer_table_structure=True,# optional
    #extract_image_block_output_dir="path/to/save/images",  # optional - only works when ``extract_image_block_to_payload=False``
    )

tables = []
images = []
texts = []

# Save image and table elements
for i, element in enumerate(elements):
    #print(f"{i}. Type: {element.category}")
    
    if element.category == "Image":
        images.append(element.metadata.image_base64)
    
    elif element.category == "Table":
        #print("Table HTML:")
        tables.append(element.metadata.text_as_html)
    else:
        texts.append(element.text)
        
# Prompt
prompt_text = """You are an assistant tasked with summarizing tables and text. \
Give a concise summary of the table or text. Table or text chunk: {element} """
prompt = ChatPromptTemplate.from_template(prompt_text)

# Summary chain
summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

prompt_template = """You are an assistant tasked with summarizing images for retrieval.
                Remember these images could potentially contain graphs, charts or 
                tables also.
                These summaries will be embedded and used to retrieve the raw image 
                for question answering.
                Give a detailed summary of the image that is well optimized for 
                retrieval.
                Do not add additional words like Summary: etc.
             """
messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | model | StrOutputParser()

image_summaries = chain.batch(images)

vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# The storage layer for the parent documents
store = InMemoryStore()
id_key = "doc_id"

# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
)

# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
# summary_texts = [
#     Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
# ]
# retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))

# Add image summaries
img_ids = [str(uuid.uuid4()) for _ in images]
summary_img = [
    Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
]
retriever.vectorstore.add_documents(summary_img)
retriever.docstore.mset(list(zip(img_ids, images)))

# Retrieve
# docs = retriever.invoke(
#     "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
# )

def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    # context_text = ""
    # if len(docs_by_type["texts"]) > 0:
    #     for text_element in docs_by_type["texts"]:
    #         context_text += text_element.text
    
    
    # construct prompt with context (including images)
    prompt_template = f"""You are an analyst tasked with understanding detailed information 
                and trends from text documents,
                data tables, and charts and graphs in images.
                You will be given context information below which will be a mix of 
                text, tables, and images usually of charts or graphs.
                Use this information to provide answers related to the user 
                question.
                Do not make up answers, use the provided context documents below and 
                answer the question to the best of your ability.
                
                User question:
                {user_question}
                
                Context documents:
                {docs_by_type}
                
                Answer:
            """

    prompt_content = [{"type": "text", "text": prompt_template}]

    

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


chain_with_rag = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | model
    | StrOutputParser()
)
chain_with_sources = {
    "context": retriever,
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
)

response = chain_with_sources.invoke(
    "Tell me detailed statistics of the top 5 years with largest wildfire acres burned"
)

print("############################### RESPONSE ###############################")
print(response)