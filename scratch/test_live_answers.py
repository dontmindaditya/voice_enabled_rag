import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Ensure BENCHMARK_MODE is NOT set
if "BENCHMARK_MODE" in os.environ:
    del os.environ["BENCHMARK_MODE"]

from src.rag.generator import GroqGenerator
from src.rag.retriever import LanceRetriever

async def test_query(generator, retriever, query):
    print(f"\nQuery: '{query}'")
    ret = retriever.retrieve(query)
    print(f"Top Retrieved Context: {repr(ret['context'][:150])}...")
    res = await generator.generate_grounded_answer(query, ret['context'])
    print(f"Answer: '{res['answer']}' (Latency: {res['latency_ms']} ms)")

async def main():
    generator = GroqGenerator()
    retriever = LanceRetriever()
    
    await test_query(generator, retriever, "what is the national language of india")
    await test_query(generator, retriever, "How many states are there in India?")

if __name__ == "__main__":
    asyncio.run(main())
