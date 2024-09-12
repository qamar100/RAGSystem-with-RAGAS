import os
from functools import lru_cache
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from llama_index.core.node_parser import MarkdownElementNodeParser
from openai import AsyncOpenAI
import asyncio
import aiohttp
from fastapi.responses import StreamingResponse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.memory import ConversationBufferWindowMemory


# Environment variable setup
api_key = os.getenv("OPENAI_API_KEY")
lamacloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY") #used for  parsing the documet
# FastAPI setup
app = FastAPI()


class Query(BaseModel):
    text: str

llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Persistent storage path
STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Load or create the index
try:
    # Load or create the storage context
    storage_context = StorageContext.from_defaults(
        persist_dir=STORAGE_DIR
    )
    print("Storage context created successfully.")
    index = load_index_from_storage(storage_context)
    print("Existing index loaded successfully.")
    
except Exception as e:
    print(f"Failed to create storage context: {e}")
    
    print("No existing index found. Creating a new index...")
    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        language="en",
        num_workers=2,
    )
    file_path = r"PATH_TO_YOUR_FILE"
    

    documents = parser.load_data(file_path)
    if not documents:
        print("No content loaded. Please check the file path and format.")
    else:
        print("File content successfully loaded.")
    
    node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)
    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    
    index = VectorStoreIndex(nodes=base_nodes + objects)
    index.storage_context.persist()
    
    print("New index created and persisted.")

# Set up ConversationBufferMemory
memory = ConversationBufferWindowMemory(k=5)

conversation = ConversationChain(
    llm=LangChainOpenAI(temperature=0),
    memory=memory,
    verbose=True
)

def get_context(query_text):
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query_text)
    context = "\n".join([node.get_content() for node in nodes])
    return context



def generate_rag_prompt(query, context):
    return f"""You are an AI assistant specializing answer general user queries. Your role is to provide accurate, helpful, and friendly assistance to the user. Always maintain a professional and empathetic tone.
Use the provided context to answer questions accurately. If the context doesn't cover a question, say: "I apologize, but I don't have specific information about that in my current database.

Context: {context}

Query: {query}

Please provide a detailed, thoughtful, response based on the context provided and the conversation history.
    """
query_engine = index.as_query_engine(
    similarity_top_k=3,
    cache=True,
    streaming=True
)
# Thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=4)



# Async HTTP client session
async def get_aiohttp_session():
    return aiohttp.ClientSession()

@lru_cache(maxsize=1000)
def get_cached_context(query_text: str) -> str:
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve(query_text)
    return "\n".join([node.get_content() for node in nodes])

async def async_query(query_text: str, session: aiohttp.ClientSession) -> str:
    context = await asyncio.to_thread(get_cached_context, query_text)
    
    # Get conversation history
    conversation_history = memory.buffer
    
    rag_prompt = generate_rag_prompt(query_text, context, conversation_history)
    
    async def stream_response():
        response = await asyncio.to_thread(query_engine.query, rag_prompt)
        full_response = ""
        for token in response.response_gen:
            full_response += token
            yield token
        
        # Update conversation memory
        conversation.predict(input=query_text)
        memory.save_context({"input": query_text}, {"output": full_response})

    return stream_response()

@app.post("/query")
async def query(query: Query):
    session = await get_aiohttp_session()
    async def generate():
        async for token in await async_query(query.text, session):
            yield token
    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)