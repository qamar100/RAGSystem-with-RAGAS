import json
import asyncio
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_utilization,
    context_recall,
)
from datasets import Dataset
import os
# Existing imports from your RAG system
from langchain.llms import OpenAI
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from RAGsystem import generate_rag_prompt

# Environment variable setup
api_key = os.getenv("OPENAI_API_KEY")
lamacloud_api_key = os.getenv("LLAMA_CLOUD_API_KEY") #used for  parsing the documet


def load_eval_dataset(file_path):
       with open(file_path, 'r') as file:
           data = json.load(file)
       return data


eval_dataset = load_eval_dataset(r'YOUR_EVAL_DATASET_PATH.json')

# Initialize OpenAI and embedding models
# LlamaIndex setup
llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.llm = llm
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

STORAGE_DIR = "./storage"
storage_context = StorageContext.from_defaults(
    persist_dir=STORAGE_DIR
)
print("Storage context created successfully.")
index = load_index_from_storage(storage_context)
print("Existing index loaded successfully.")


# Set up your query engine
query_engine = index.as_query_engine(similarity_top_k=3)

async def async_query(query_text):
    response = await asyncio.to_thread(query_engine.query, query_text)
    return response.response

def get_context(query_text):
    retriever = index.as_retriever(similarity_top_k=2)
    nodes = retriever.retrieve(query_text)
    return "\n".join([node.get_content() for node in nodes])


async def rag_integrated_query(question):
       context = get_context(question)  # Your existing context retrieval function
       rag_prompt = generate_rag_prompt(question, context)  # Your existing RAG prompt generation function
       response = await async_query(rag_prompt)  # Your existing query function
       return response, context
   
print("Starting evaluation...")

def prepare_ragas_data(eval_dataset):
        ragas_data = []
        for i, item in enumerate(eval_dataset):
            question = item["question"]
            ground_truths = item["ground_truths"]
            rag_answer = asyncio.run(async_query(question))
            context = get_context(question)
            
            print(f"Processing item {i+1}/{len(eval_dataset)}")
            print(f"Question: {question}")
            print(f"RAG Answer: {rag_answer[:150]}...")  # Print first 100 chars
            print(f"Context (first 100 chars): {context[:150]}...")
            print("---")
            
            ragas_data.append({
                "question": question,
                "answer": rag_answer,
                "ground_truths": ground_truths,  # This matches your dataset structure the second part is the table that i have in my dataset
                "contexts": [context],
                "ground_truth": ground_truths[0]  # Add this line to satisfy RAGAS
            })
        return ragas_data


ragas_data = prepare_ragas_data(eval_dataset)

# Convert to Dataset format
dataset = Dataset.from_list(ragas_data)

required_columns = ['question', 'answer', 'contexts', 'ground_truths', 'ground_truth']
for col in required_columns:
    if col not in dataset.column_names:
        raise ValueError(f"Required column '{col}' is missing from the dataset")

# Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_utilization,
    ]
)

print("-------------Evaluation complete.------------")

print("--------------Analyzing context--------------")

def analyze_contexts(eval_dataset, ragas_data):
       for eval_item, ragas_item in zip(eval_dataset, ragas_data):
           question = eval_item["question"]
           ground_truth = eval_item["ground_truths"][0]
           retrieved_context = ragas_item["contexts"][0]
           
           print(f"Question: {question}")
           print(f"Ground Truth: {ground_truth[:100]}...")
           print(f"Retrieved Context: {retrieved_context[:100]}...")
           print("---")
# After preparing ragas_data
analyze_contexts(eval_dataset, ragas_data)



print(result)

print(f"Faithfulness: {result['faithfulness']}")
print(f"Answer Relevancy: {result['answer_relevancy']}")
print(f"Context Recall: {result['context_recall']}")
print(f"context utilization: {result['context_utilization']}")