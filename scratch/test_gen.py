import asyncio
import os
import time
from dotenv import load_dotenv
load_dotenv()

from src.rag.generator import GroqGenerator
from src.rag.retriever import LanceRetriever

async def test():
    generator = GroqGenerator()
    # Try llama-3.2-3b-preview
    generator.model = "llama-3.2-3b-preview"
    
    retriever = LanceRetriever(db_path="./vector_db/msmarco.lancedb", table_name="msmarco_chunks")
    
    query = "What is the capital of Goa?"
    ret_res = retriever.retrieve(query)
    context = ret_res["context"]
    
    print("--- Running queries with llama-3.2-3b-preview ---")
    for i in range(3):
        res = await generator.generate_grounded_answer(query, context)
        print(f"Query {i+1} Answer: {res['answer']}")
        print(f"Query {i+1} Latency: {res['latency_ms']} ms")
        await asyncio.sleep(0.05)

if __name__ == "__main__":
    asyncio.run(test())
